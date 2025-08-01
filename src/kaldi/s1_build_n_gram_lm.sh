#!/bin/bash
. ./path.sh
. ./cmd.sh
. ./configs.sh
###############

lm_dir=/data/asr/kaldi/exp/lm
out_dir=$lm_dir/
mkdir $out_dir 
corpus=$lm_dir/llm_corpus.txt
echo "Step 02.03: Build Language model for training"
if [ ! -f $out_dir/lm_test.arpa.gz ]; then
	ngram-count -order 3 -unk \
		-ndiscount \
		-interpolate \
		-limit-vocab \
		-text "${corpus}" \
		-lm $out_dir/lm_train.arpa.gz  \
		-write-vocab  $out_dir/vocab 
fi
echo "Step 02: Done!!!"
