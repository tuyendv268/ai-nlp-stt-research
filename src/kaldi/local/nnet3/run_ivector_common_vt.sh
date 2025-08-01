#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true

. ./path.sh
. ./utils/parse_options.sh
dir=expTDNN2/nnet3
mkdir -p ${dir}
# perturbed data preparation
train_set=data/mfcc_hires/train_vin_aug
train_set_350k=data/mfcc_hires/train_vin_aug
train_set_250k=data/mfcc_hires/train_vin_aug
ali_dir_2000k_nudup=expmfcc_pitch/tri5b_ali_aug_mfcc_hires
lang=data/lang_nosp_withLM_training_Training-dict-OnlyBac-XSAMPA
#########################

if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 12 \
    --splice-opts "--left-context=3 --right-context=3" \
    4000 80000 ${train_set_250k} \
    ${lang} ${ali_dir_2000k_nudup} ${dir}/tri5b
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj --num-frames 500000 \
    ${train_set_250k} 512 ${dir}/tri5b ${dir}/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $ivec_job \
    ${train_set_350k} ${dir}/diag_ubm ${dir}/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 3 ${train_set} ${train_set}_max3
  utils/data/fix_data_dir.sh ${train_set}_max3
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${train_set}_max3 ${dir}/extractor ${dir}/ivectors_$(basename $train_set)_max3 || exit 1;
fi
