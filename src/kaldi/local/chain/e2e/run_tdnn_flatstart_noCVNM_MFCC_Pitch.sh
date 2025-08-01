#!/bin/bash
# Copyright    2017  Hossein Hadian

# This script performs chain training in a flat-start manner
# and without building or using any context-dependency tree.
# It does not use ivecors or other forms of speaker adaptation
# except simple mean and variance normalization.
# It is called from run_e2e_phone.sh

# Note: this script is configured as phone-based, if you want
# to run it in character mode, you'll need to change _nosp
# to _char everywhere and also copy char_lm.fst instead
# of phone_lm.fst (in stage 1 below)

# local/chain/compare_wer.sh exp/chain/e2e_tdnn_1a
# System                e2e_tdnn_1a
#WER dev93 (tgpr)                9.70
#WER dev93 (tg)                  9.05
#WER dev93 (big-dict,tgpr)       7.20
#WER dev93 (big-dict,fg)         6.36
#WER eval92 (tgpr)               5.88
#WER eval92 (tg)                 5.32
#WER eval92 (big-dict,tgpr)      3.67
#WER eval92 (big-dict,fg)        3.05
# Final train prob        -0.0741
# Final valid prob        -0.0951
# Final train prob (xent)
# Final valid prob (xent)
# Num-params                 5562234

# steps/info/chain_dir_info.pl exp/chain/e2e_tdnn_1a
# exp/chain/e2e_tdnn_1a: num-iters=68 nj=2..5 num-params=5.6M dim=40->84 combine=-0.094->-0.094 logprob:train/valid[44,67,final]=(-0.083,-0.073,-0.072/-0.097,-0.095,-0.095)

set -e

# configs for 'chain'
stage=3
train_stage=0 # -10
get_egs_stage=-10 # -10
affix=NoCVMN_MFCC40

# training options
num_epochs=10
num_jobs_initial=1
num_jobs_final=2
minibatch_size=8,16,32,64
common_egs_dir=
l2_regularize=0.00005
frames_per_iter=3000000
# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.3@0.50,0'

cmvn_opts="--norm-means=false --norm-vars=false"
#cmvn_opts=""
#train_set=data/mfcc_hires/training_full_cleaned_noseg_nonoise_ok
train_set=data/mfcc_hires/train_aug2xR_And_Org_3speed
#test_sets="test_dev93 test_eval92"

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

lang=data/lang_e2e
langin=data/lang_nosp_withLM_training_Training-dict-OnlyBac-XSAMPA
langtest=data/lang_nosp_withLM_testing_Testing-dict-OnlyBac-XSAMPA
treedir=expE2E/chain/e2e_tree  # it's actually just a trivial tree (no tree building)
dir=expE2E/chain/e2e_cnn_tdnn_bsltm_${affix}

if [ $stage -le 0 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r ${langin} $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 1 ]; then
  steps/nnet3/chain/e2e/prepare_e2e.sh  --cmd "$train_cmd" \
                                       --shared-phones true \
                                       $train_set $lang $treedir
  cp expE2E/chain/e2e_base/phone_lm.fst $treedir/
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.01"
  
  tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"

  lstm_opts="l2-regularize=0.0025 decay-time=20 dropout-proportion=0.0"
  opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.0025"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input
   
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10  time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256

  # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
  # information bottleneck doesn't become a problem.  (we use time-stride=0 so no splicing, to
  # limit the num-parameters).
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3

  fast-lstmp-layer name=blstm1-forward input=tdnnf15 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm1-backward input=tdnnf15 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=3 $lstm_opts

  fast-lstmp-layer name=blstm2-forward input=Append(blstm1-forward, blstm1-backward) cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  fast-lstmp-layer name=blstm2-backward input=Append(blstm1-forward, blstm1-backward) cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=3 $lstm_opts

   ## adding the layers for chain branch
  output-layer name=output input=Append(blstm2-forward, blstm2-backward) output-delay=8 include-log-softmax=false dim=$num_targets max-change=1.5

  output-layer name=output-xent input=Append(blstm2-forward, blstm2-backward) output-delay=8 dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5  


EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 3 ]; then
  # no need to store the egs in a shared storage because we always
  # remove them. Anyway, it takes only 5 minutes to generate them.

  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights false \
    --chain.xent-regularize $xent_regularize \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.momentum 0 \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.00001 \
    --trainer.optimization.shrink-value 1.0 \
    --trainer.max-param-change 2.0 \
    --trainer.dropout-schedule $dropout_schedule \
    --cleanup.remove-egs false \
    --feat-dir ${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi
if [ $stage -le 4 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/mkgraph.sh \
    --self-loop-scale 1.0 ${langtest} \
    $dir $dir/graph_lang-testing-EN-Thuoc-Transcript-Vocab_Dict || exit 1;
fi

exit 0;

if [ $stage -le 5 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          $treedir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgpr,tg} \
        data/${data}_hires ${dir}/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_nosp_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

echo "Done. Date: $(date). Results:"
local/chain/compare_wer.sh $dir
