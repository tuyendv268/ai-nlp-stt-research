from transformers import pipeline
import multiprocessing as mp
import torch
import jiwer
from tqdm import tqdm
from glob import glob
import json
import os

from src.cores.asr_core import ASR_Core
import torchaudio

def split_data_to_batch(data, batch_size):
    n_samples = len(data)
    if n_samples % batch_size == 0:
        n_batches = int(len(data)/batch_size)
    else:
        n_batches = int(len(data)/batch_size) + 1

    batches = []
    for i in range(n_batches):
        batch = data[i*batch_size: (i+1)*batch_size]
        batches.append(batch)

    return batches

def load_jsonl_data(metadata_path, load_json_obj=False):
    if load_json_obj:
        metadata = list(map(
            json.loads, 
            tqdm(
                open(metadata_path).readlines(), 
                desc="Load data"
                )
            )
        )
    else:
        metadata = open(metadata_path).readlines()
    
    return metadata

def save_jsonl_data(metadata_path, metadata):
    with open(metadata_path, "w") as f:
        for sample in metadata:
            json_obj = json.dumps(sample, ensure_ascii=False)
            f.write(json_obj + "\n")
    
    print(f'###saved {len(metadata)} records to {metadata_path}')
            
    return metadata

def load_cached_metadata(metadata_dir):
    filepaths = glob(f'{metadata_dir}/*.jsonl')
    metadata = []
    for filepath in filepaths:
        for line in open(filepath).readlines():
            sample = json.loads(line)
            
            audio_filepath = sample["audio_filepath"]
            filename = os.path.basename(audio_filepath)
            filename = "-".join(filename.split("-")[:-4])
            metadata.append(filename)
            
    metadata = set(metadata)
    
    return metadata

def main(metadata, pred_metadata_path, batch_size=8):
    model_name = "finetuned_wav2vec2_large"
    asr_core = ASR_Core(
        processor_path="/data/ai-nlp-stt-service/exp/ckpts/wav2vec2/processor",
        ckpt_path="/data/ai-nlp-stt-service/exp/ckpts/wav2vec2/model_epoch5_step_final.pt",
        config_path="/data/ai-nlp-stt-service/exp/ckpts/wav2vec2/config.json",
    )
        
    with open(pred_metadata_path, "a+") as f:
        metadata = sorted(metadata, key=lambda x: x["duration"])
        batches = split_data_to_batch(metadata, batch_size=batch_size)
        for index in tqdm(
            range(len(batches)),
            desc="Infer",
            postfix=f"pid={os.getpid()}"
        ):
            batch = batches[index]
            
            audio_filepaths = list(map(lambda x: x["audio_filepath"], batch))
            audio_filepaths = [i for i in audio_filepaths if os.path.exists(i)]
            if len(audio_filepaths) == 0:
                continue
            
            #####
            audio_batch = [
                torchaudio.load(filepath)[0][0]
                for filepath in audio_filepaths
            ]
            preds = asr_core.run(audio_batch)
            #####
            
            if index % 5 == 0:
                torch.cuda.empty_cache()
            
            for i in range(len(audio_filepaths)):
                sample = batch[i]
                # pred = preds[i]["text"]
                pred = preds[i]
                
                if len(pred.strip()) <= 0:
                    continue

                audio_filepath = sample["audio_filepath"]
                duration = sample["duration"]
                
                pred = pred.lower()

                sample = {
                    "audio_filepath": audio_filepath,
                    "duration": duration,
                    "pred": pred,
                    "model_name": model_name
                }
                json_obj = json.dumps(sample, ensure_ascii=False)
                f.write(json_obj+"\n")

    print(f'###done !!!')

if __name__ == "__main__":        
    filepaths = [
        # "/data/asr-research/data/vad_metadata/vad_metadata-0.jsonl",
        "/data/asr-research/data/vad_metadata/vad_metadata-1.jsonl",
    ]
    pred_metadata_dir = "/data/asr-research/data/stt_metadata_w2v2_part_2"
    
    metadata_dir = "/data/asr-research/data/stt_metadata_w2v2"
    cached_filenames = load_cached_metadata(metadata_dir)

    if not os.path.exists(pred_metadata_dir):
        os.mkdir(pred_metadata_dir)

    for metadata_path in filepaths:
        filename = os.path.basename(metadata_path)
        filename = filename.split(".")[0]

        merged_pred_metadata_path = f'{pred_metadata_dir}/pred-{filename}.jsonl'

        # load metadata
        metadata = load_jsonl_data(metadata_path, load_json_obj=True)
        # print(f'number of sample: {len(metadata)}')
        # metadata = [
        #     sample for sample in tqdm(metadata, desc="Filter data")
        #     if sample["orig_filename"] not in cached_filenames
        # ]
        # print(f'number of sample: {len(metadata)}')

        num_workers = 6
        batch_size = int(round(len(metadata) / num_workers) + 1)
        params = split_data_to_batch(metadata, batch_size=batch_size)
        params = [
            (param, f'{pred_metadata_dir}/pred-{filename}-{index}.jsonl') 
            for index, param in enumerate(params)
        ]
        
        # run inference
        with mp.Pool(processes=num_workers) as pool:
            pool.starmap(func=main, iterable=params) 
            
        # # merge metadata
        # metadata = []
        # filepaths = [param[1] for param in params]
        # for filepath in filepaths:
        #     sub_metadata = load_jsonl_data(metadata_path=filepath, load_json_obj=True)
        #     metadata += sub_metadata
        #     os.remove(filepath)
        #     print(f'###remove {filepath}')
            
        # # save metadata
        # save_jsonl_data(
        #     metadata_path=merged_pred_metadata_path,
        #     metadata=metadata
        # )