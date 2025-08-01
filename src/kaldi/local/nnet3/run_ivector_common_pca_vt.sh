#!/bin/bash

set -e -o pipefail


# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.


stage=0
num_threads_ubm=16
num_processes=4
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

DNN_dir=$1 #
train_set=$2    # MFCC-Hires data
oneof=$3 # taking only 1/oneof cho UBM and PCA training
gmm_gauss=$4 # Number of Gaussian mixtures for UBM model

if [ $stage -le 4 ]; then
  echo "$0: making a subset of data to train the diagonal UBM and the PCA transform."  
  mkdir -p ${DNN_dir}/diag_ubm
  temp_data_root=${DNN_dir}/diag_ubm 
  # We'll 1/20 of the data, since the Data is very large.
  num_utts_total=$(wc -l ${train_set}/utt2spk | awk '{print $1}')
  num_utts=$[$num_utts_total/$oneof]
  utils/data/subset_data_dir.sh ${train_set} \
     $num_utts ${temp_data_root}/$(basename ${train_set})_subset || exit 1;

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 500000 --subsample 2 \
       ${temp_data_root}/$(basename ${train_set})_subset \
       ${DNN_dir}/pca_transform || exit 1;

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 32 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/$(basename ${train_set})_subset $gmm_gauss \
    ${DNN_dir}/pca_transform ${DNN_dir}/diag_ubm || exit 1;
fi

if [ $stage -le 5 ]; then
  # iVector extractors can in general be sensitive to the amount of data, but
  # this one has a fairly small dim (defaults to 100) so we don't use all of it,
  # we use just the 60k subset (about one fifth of the data, or 200 hours).
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 --num-processes $num_processes \
    ${temp_data_root}/$(basename ${train_set})_subset ${DNN_dir}/diag_ubm ${DNN_dir}/extractor || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: extracting iVectors for training data"
  ivectordir=${DNN_dir}/ivectors_$(basename ${train_set})
  
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker. this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${train_set} ${ivectordir}_max2 || exit 1;

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    ${ivectordir}_max2 ${DNN_dir}/extractor \
    $ivectordir || exit 1;
fi
echo "Done extracting I-vector, output at: ${ivectordir}"
exit 0;
