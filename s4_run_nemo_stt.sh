#!/bin/bash

export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

lm_ckpt_path="/data/asr/vi_lm_5grams.bin"
asr_ckpt_path="/data/asr/nemo_experiments/FastConformer-CTC-BPE/2025-07-28_15-26-01/checkpoints/FastConformer-CTC-BPE--val_wer--0.1891.ckpt"
manifest_path=[\
"/data/asr/metadata/pred-train_gigaspeech2_filtered.jsonl"\
]
output_path=data/s10_stt_metadata_nemo_ctc
batch_size=128

python s4_run_nemo_stt.py \
    asr_ckpt_path=$asr_ckpt_path \
    lm_ckpt_path=$lm_ckpt_path \
    predict_ds.manifest_filepath=$manifest_path \
    predict_ds.batch_size=$batch_size \
    output_path=$output_path