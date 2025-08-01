import itertools
import json
import os
from dataclasses import dataclass, field, is_dataclass
from typing import Optional

import lightning.pytorch as ptl
import torch
from omegaconf import MISSING, OmegaConf

from nemo.collections.asr.data.audio_to_text_dataset import ASRPredictionWriter
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.models.configs import ASRDatasetConfig
from nemo.core.config import TrainerConfig, hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@dataclass
class ParallelTranscriptionConfig:
    asr_ckpt_path: Optional[str] = None  # name
    lm_ckpt_path: Optional[str] = None  # name

    predict_ds: ASRDatasetConfig = field(
        default_factory=lambda: ASRDatasetConfig(
            return_sample_id=True, 
            num_workers=4, 
            min_duration=0, 
            max_duration=40
        )
    )
    output_path: str = MISSING

    # when return_predictions is enabled, the prediction call would keep all the predictions in memory and return them when prediction is done
    return_predictions: bool = False

    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(devices=-1, accelerator="gpu", strategy="ddp")
    )


def match_train_config(predict_ds, train_ds):
    # It copies the important configurations from the train dataset of the model
    # into the predict_ds to be used for prediction. It is needed to match the training configurations.
    if train_ds is None:
        return

    predict_ds.sample_rate = train_ds.get("sample_rate", 16000)
    cfg_name_list = [
        "int_values",
        "use_start_end_token",
        "blank_index",
        "unk_index",
        "normalize",
        "parser",
        "eos_id",
        "bos_id",
        "pad_id",
    ]

    if is_dataclass(predict_ds):
        predict_ds = OmegaConf.structured(predict_ds)
    for cfg_name in cfg_name_list:
        if hasattr(train_ds, cfg_name):
            setattr(predict_ds, cfg_name, getattr(train_ds, cfg_name))

    return predict_ds

def init_asr_model_with_lm(asr_ckpt_path, lm_ckpt_path):
    from dataclasses import dataclass, field
    from typing import List

    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
    from nemo.collections.asr.parts.submodules import ctc_beam_decoding
    
    @dataclass
    class BeamSearchNGramConfig:
        decoding_mode: str = "beamsearch_ngram"

        beam_width: List[int] = field(default_factory=lambda: [32])
        beam_alpha: List[float] = field(default_factory=lambda: [1.0])
        beam_beta: List[float] = field(default_factory=lambda: [0.0])

        decoding_strategy: str = "pyctcdecode"
        decoding: ctc_beam_decoding.BeamCTCInferConfig = field(
            default_factory=lambda: ctc_beam_decoding.BeamCTCInferConfig(beam_size=128)
        )

    # change decoding config
    cfg = BeamSearchNGramConfig()

    cfg.decoding.beam_size = 32
    cfg.decoding.beam_alpha = 0.4
    cfg.decoding.beam_beta = 1.5
    cfg.decoding.return_best_hypothesis = True
    cfg.decoding.kenlm_path = lm_ckpt_path
    
    # init asr model
    model = EncDecCTCModelBPE.load_from_checkpoint(asr_ckpt_path)
    model.change_decoding_strategy(None)

    model.cfg.decoding.strategy = cfg.decoding_strategy
    model.cfg.decoding.beam = cfg.decoding

    model.change_decoding_strategy(model.cfg.decoding)

    return model

@hydra_runner(config_name="TranscriptionConfig", schema=ParallelTranscriptionConfig)
def main(cfg: ParallelTranscriptionConfig):
    model = init_asr_model_with_lm(cfg.asr_ckpt_path, cfg.lm_ckpt_path)

    cfg.predict_ds.return_sample_id = True
    cfg.predict_ds = match_train_config(
        predict_ds=cfg.predict_ds, 
        train_ds=model.cfg.train_ds
    )

    if cfg.predict_ds.use_lhotse:
        OmegaConf.set_struct(cfg.predict_ds, False)
        cfg.trainer.use_distributed_sampler = False
        cfg.predict_ds.force_finite = True
        cfg.predict_ds.force_map_dataset = True
        cfg.predict_ds.do_transcribe = True
        OmegaConf.set_struct(cfg.predict_ds, True)

    trainer = ptl.Trainer(**cfg.trainer)

    if cfg.predict_ds.use_lhotse:
        OmegaConf.set_struct(cfg.predict_ds, False)
        cfg.predict_ds.global_rank = trainer.global_rank
        cfg.predict_ds.world_size = trainer.world_size
        OmegaConf.set_struct(cfg.predict_ds, True)

    data_loader = model._setup_dataloader_from_config(cfg.predict_ds)

    os.makedirs(cfg.output_path, exist_ok=True)
    # trainer.global_rank is not valid before predict() is called. Need this hack to find the correct global_rank.
    global_rank = trainer.node_rank * trainer.num_devices + int(os.environ.get("LOCAL_RANK", 0))
    output_file = os.path.join(cfg.output_path, f"predictions_{global_rank}.jsonl")
    predictor_writer = ASRPredictionWriter(dataset=data_loader.dataset, output_file=output_file)
    trainer.callbacks.extend([predictor_writer])

    predictions = trainer.predict(model=model, dataloaders=data_loader, return_predictions=cfg.return_predictions)
    if predictions is not None:
        predictions = list(itertools.chain.from_iterable(predictions))
    samples_num = predictor_writer.close_output_file()

    logging.info(
        f"Prediction on rank {global_rank} is done for {samples_num} samples and results are stored in {output_file}."
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    samples_num = 0
    pred_text_list = []
    if is_global_rank_zero():
        output_file = os.path.join(cfg.output_path, f"predictions_all.jsonl")
        logging.info(f"Prediction files are being aggregated in {output_file}.")
        with open(output_file, 'w') as outf:
            for rank in range(trainer.world_size):
                input_file = os.path.join(cfg.output_path, f"predictions_{rank}.jsonl")
                with open(input_file, 'r') as inpf:
                    lines = inpf.readlines()
                    for line in lines:
                        item = json.loads(line)
                        pred_text_list.append(item["pred_text"])
                        outf.write(json.dumps(item, ensure_ascii=False) + "\n")
                        samples_num += 1
        logging.info(
            f"Prediction is done for {samples_num} samples in total on all workers and results are aggregated in {output_file}."
        )


if __name__ == '__main__':
    main()