#!/bin/bash

# Copyright   2016  Vimal Manohar
#             2016  Johns Hopkins University (author: Daniel Povey)
#             2017  Nagendra Kumar Goel
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh

# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets
# [will add these later].

set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
data=$1
cleanup_affix=cleaned
srcdir=$2
langdir=data/lang
decode_nj=8
decode_num_threads=2

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh

cleaned_data=${data}_${cleanup_affix}

dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data_DNN_Lats.sh --stage $cleanup_stage \
    --nj $nj --cmd "$train_cmd" \
    $data $langdir $srcdir $dir $cleaned_data
fi
exit 0;