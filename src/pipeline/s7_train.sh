# sudo docker run --rm -it --shm-size 8g \
#     --gpus all \
#     -v /data:/data \
#     -v /data2:/data2 \
#     --entrypoint /bin/bash g/stt-service:1.1.1

export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
# export TEMP=/home/tuyendv/tmp

# python ./process_asr_text_tokenizer.py \
#   --manifest=["/data/asr/metadata/pred-train_gigaspeech2_filtered.jsonl","/data/asr-research/f88_training.jsonl"] \
#   --data_root="tokenizers/gvi/" \
#   --vocab_size=2048 \
#   --tokenizer="spe" \
#   --spe_type="bpe" \
#   --log
#######################################################################################################

# batch_size=48
# config_path="configs"
# config_name="fastconformer_hybrid_transducer_ctc_bpe"

# # train_manifest_path=[\
# # "/data/asr-research/f88_wer_0.08.jsonl",\
# # "/data/asr-research/f88_segments_wer_0.08.jsonl"\
# # ]

# train_manifest_path=[\
# "/data/asr-research/data/metadata/merged_and_cleaned_f88_data.jsonl",\
# "/data/asr-research/f88_segments_wer_0.08.jsonl"\
# ]

# val_manifest_path=["metadata/test_tele.jsonl"]
# test_manifest_path=["metadata/test_tele.jsonl"]

# tokenizer_dir="tokenizers/gvi/tokenizer_spe_bpe_v1024"
# tokenizer_type="bpe"
# init_from_nemo_model="/data/asr/nemo_experiments/FastConformer-Hybrid-Transducer-CTC-BPE/2025-07-19_16-37-12/checkpoints/FastConformer-Hybrid-Transducer-CTC-BPE.nemo"

# python speech_to_text_hybrid_rnnt_ctc_bpe.py \
#     --config-path=$config_path \
#     --config-name=$config_name \
#     +init_from_nemo_model=$init_from_nemo_model \
#     model.train_ds.manifest_filepath=$train_manifest_path \
#     model.train_ds.batch_size=$batch_size \
#     model.test_ds.manifest_filepath=$test_manifest_path \
#     model.validation_ds.manifest_filepath=$test_manifest_path \
#     model.tokenizer.dir=$tokenizer_dir \
#     model.tokenizer.type=$tokenizer_type \
#     trainer.val_check_interval=0.125

#######################################################################################################
batch_size=48
config_path="configs"
config_name="fastconformer_ctc_bpe"

# train_manifest_path=[\
# "/data/asr-research/f88_wer_0.08.jsonl",\
# "/data/asr-research/f88_segments_wer_0.08.jsonl"\
# ]
train_manifest_path=[\
"/data/asr-research/data/metadata/f88_segments_wer_0.05_v1.jsonl",\
"/data/asr-research/data/metadata/f88_wer_0.05_v1.jsonl"\
]

val_manifest_path=["metadata/test_tele.jsonl"]
test_manifest_path=["metadata/test_tele.jsonl"]

tokenizer_dir="tokenizers/gvi/tokenizer_spe_bpe_v1024"
tokenizer_type="bpe"

init_from_nemo_model="/data/asr/nemo_experiments/FastConformer-CTC-BPE/FastConformer-CTC-BPE.nemo"

python speech_to_text_ctc_bpe.py \
    --config-path=$config_path \
    --config-name=$config_name \
    +init_from_nemo_model=$init_from_nemo_model \
    model.train_ds.manifest_filepath=$train_manifest_path \
    model.train_ds.batch_size=$batch_size \
    model.test_ds.manifest_filepath=$test_manifest_path \
    model.validation_ds.manifest_filepath=$test_manifest_path \
    model.tokenizer.dir=$tokenizer_dir \
    model.tokenizer.type=$tokenizer_type \
    trainer.val_check_interval=0.075
#######################################################################################################

# batch_size=32
# config_path="configs"
# config_name="fastconformer_ctc_bpe_xlarge"

# # train_manifest_path=[\
# # "/data/asr-research/data/metadata/f88_segments_wer_0.05_v1.jsonl",\
# # "/data/asr-research/data/metadata/f88_wer_0.05_v1.jsonl",\
# # "/data/asr-research/data/metadata/vlsp250h.jsonl"\
# # ]

# train_manifest_path=[\
# "/data/asr-research/data/metadata/f88_segments_wer_0.08.jsonl",\
# "/data/asr-research/data/metadata/f88_wer_0.08.jsonl"\
# ]

# val_manifest_path=["metadata/test_tele.jsonl"]
# test_manifest_path=["metadata/test_tele.jsonl"]

# tokenizer_dir="tokenizers/gvi/tokenizer_spe_bpe_v2048"
# tokenizer_type="bpe"
# # init_from_nemo_model="/data/asr/nemo_experiments/FastConformer-CTC-BPE-Large/2025-07-24_22-20-37/checkpoints/FastConformer-CTC-BPE-Large.nemo"
# init_from_ptl_ckpt="/data/asr/nemo_experiments/FastConformer-CTC-BPE-Large/2025-07-25_03-45-49/checkpoints/FastConformer-CTC-BPE-Large--0.2781.ckpt"
# python speech_to_text_ctc_bpe.py \
#     --config-path=$config_path \
#     --config-name=$config_name \
#     +init_from_ptl_ckpt=$init_from_ptl_ckpt \
#     model.train_ds.manifest_filepath=$train_manifest_path \
#     model.train_ds.batch_size=$batch_size \
#     model.test_ds.manifest_filepath=$test_manifest_path \
#     model.validation_ds.manifest_filepath=$val_manifest_path \
#     model.tokenizer.dir=$tokenizer_dir \
#     model.tokenizer.type=$tokenizer_type \
#     trainer.val_check_interval=0.075 \
#     trainer.accumulate_grad_batches=1

#######################################################################################################