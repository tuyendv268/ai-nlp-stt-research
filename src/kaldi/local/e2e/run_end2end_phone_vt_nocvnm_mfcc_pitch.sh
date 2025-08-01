#!/bin/bash
# Copyright 2017    Hossein Hadian

# This top-level script demonstrates end-to-end LF-MMI training (specifically
# single-stage flat-start LF-MMI models) on WSJ. It is basically like
# "../run.sh" except it does not train any GMM or SGMM models and after
# doing data/dict preparation and feature extraction goes straight to
# flat-start chain training.
# It uses a phoneme-based lexicon just like "../run.sh" does.

set -euo pipefail


stage=5
inda=
trainset=
trainset_hires=data/mfcc_hires/train_aug2xR_And_Org
lang=data/lang_nosp_withLM_training_Training-dict-OnlyBac-XSAMPA
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh
. utils/parse_options.sh


# This is just like stage 0 in run.sh except we do mfcc extraction later
# We use the same suffixes as in run.sh (i.e. _nosp) for consistency

if [ $stage -le 4 ]; then
  echo "$0: estimating phone language model for the denominator graph"
  mkdir -p expE2E/chain/e2e_base/log
  $train_cmd expE2E/chain/e2e_base/log/make_phone_lm.log \
  cat ${trainset_hires}/text \| \
    steps/nnet3/chain/e2e/text_to_phones.py ${lang} \| \
    utils/sym2int.pl -f 2- ${lang}/phones.txt \| \
    chain-est-phone-lm --num-extra-lm-states=2000 \
                       ark:- expE2E/chain/e2e_base/phone_lm.fst
fi

if [ $stage -le 5 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_flatstart_noCVNM_MFCC_Pitch.sh
fi
