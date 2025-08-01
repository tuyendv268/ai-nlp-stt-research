#!/bin/bash

DIRECTORY="/data/asr/kaldi"

find "$DIRECTORY" -type f -exec chmod +x {} \;
echo "Execute permissions added to all files in $DIRECTORY and its subdirectories"

sudo docker run -it --rm --gpus=all \
    -v /data/asr-research/src/kaldi:/data/asr-research/src/kaldi \
    -v /data2/audio/f88-v1:/data2/audio/f88-v1 \
    -v /data/asr/asr_data:/data/asr/asr_data \
    g/kaldi:22.10-py3 /bin/bash 