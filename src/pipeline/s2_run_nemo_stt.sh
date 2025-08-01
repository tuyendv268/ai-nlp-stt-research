#!/bin/bash

export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

lm_ckpt_path="/data/asr/vi_lm_5grams.bin"
asr_ckpt_path="/data/asr/nemo_experiments/FastConformer-CTC-BPE/2025-07-21_09-10-56/checkpoints/FastConformer-CTC-BPE--0.1946.ckpt"
manifest_path=[\
"/data/asr-research/data/s1_vad_metadata/vad_metadata-0.jsonl",\
"/data/asr-research/data/s1_vad_metadata/vad_metadata-1.jsonl"\
]
output_path=data/s2_stt_metadata_nemo_ctc
batch_size=128

python s2_run_nemo_stt.py \
    asr_ckpt_path=$asr_ckpt_path \
    lm_ckpt_path=$lm_ckpt_path \
    predict_ds.manifest_filepath=$manifest_path \
    predict_ds.batch_size=$batch_size \
    output_path=$output_path