# ## Install dependencies
# !pip install wget
# !pip install text-unidecode
# !pip install matplotlib>=3.3.2

# ## Install NeMo
# !python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]

# python process_asr_text_tokenizer.py \
#   --manifest="/data/asr/metadata/pred-train_gigaspeech2_filtered.jsonl" \
#   --data_root="tokenizers/gvi/" \
#   --vocab_size=4096 \
#   --tokenizer="spe" \
#   --spe_type="bpe" \
#   --log

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_hybrid_rnnt_ctc_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    model.aux_ctc.ctc_loss_weight=0.3 \
    trainer.devices=-1 \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

"""

from omegaconf import OmegaConf
import nemo.lightning as nl

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils.trainer_utils import resolve_trainer_cfg
from nemo.utils.exp_manager import exp_manager
from nemo.core.config import hydra_runner
from nemo.utils import logging

@hydra_runner(config_path="../conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = nl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)

if __name__ == '__main__':
    main()

