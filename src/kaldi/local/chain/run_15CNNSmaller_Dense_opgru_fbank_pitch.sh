#!/bin/bash
set -e

# configs for 'chain'
# stage=0
num_leaves=5000
train_stage=-10
get_egs_stage=-10

# TDNN options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

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

#################
dir=$1 # Working DNN dir for this setup
lang=$2 # DNN lang topo (1 stage HMM)
langtrain=$3 # Reference Lang for DNN training, should be the HMM-GMM lang training.
train_set_for_tree=$4 #Data used that associates with the input Alignments
gmm_dir=$5 # GMM-HMM model dir for tree tranining
ali_dir=$6 # Alignment dir for tree training
lat_dir=$7 # Lattices dir that have to associate with train_data_dir (MFCC-Hires)
train_data_dir=$8 # The input data for DNN training, should be argumented data in kind of MFCC-Hires feature
train_ivector_dir=$9 #The i-vector data that associate with train_data_dir (MFCC-Hires), and extracted from script 10.Extract_ivector_pca
extractor=${10} # The i-vector extractor
train_stage=${11}
get_egs_stage=${12}
stage=${13}
num_epoch=${14} # Number of epoches for DNN training (should be more than 3)
init_num_jobs=${15} # Number of paralell jobs at begining of DNN training 
final_num_jobs=${16} # Number of paralell jobs at lats epoch of DNN training. Careful on GPU-Mem = final_num_jobs * size_of_one_job
minibatch=${17} # minibatch sizes, like: 128,64
init_LrRate=${18} # initial learning rate
final_LrRate=${19} # final learning rate
trained_model=${20} # pre-trained model that have to be the same topo, set to None to skip it
cuda_devices=${21}
#################
tree_dir=$(dirname ${dir})/chain/$(basename ${gmm_dir})_tree

if [ $stage -le 0 ]; then
  for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
      $train_set_for_tree/feats.scp $ali_dir/ali.1.gz; do
    [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
  done
fi

if [ $stage -le 1 ]; then
  echo =============debug1================
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt ${langtrain}/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r ${langtrain} $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 2 ]; then
  # Build a tree using our new topology. We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" $num_leaves ${train_set_for_tree} $lang $ali_dir $tree_dir || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.0"
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_first_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=1.0"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig

  input dim=26 name=input 
  ### CNN MFCC
  conv-relu-batchnorm-layer name=fea_cnn1 $cnn_opts height-in=26 height-out=26 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32
  conv-relu-batchnorm-layer name=fea_cnn2 $cnn_opts height-in=26 height-out=13 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32
  conv-relu-batchnorm-layer name=fea_cnn3 $cnn_opts height-in=13 height-out=13 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32
  conv-relu-batchnorm-layer name=fea_cnn4 $cnn_opts height-in=13 height-out=12 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=fea_cnn5 $cnn_opts height-in=12 height-out=11 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=fea_cnn $cnn_opts height-in=11 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  
  # RES-1
  conv-relu-batchnorm-layer input=fea_cnn name=mfcc_cnn5 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  no-op-component name=cnn_res5 input=Sum(fea_cnn,mfcc_cnn5)
  conv-relu-batchnorm-layer input=cnn_res5 name=mfcc_cnn6 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  no-op-component name=cnn_res6 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6)
  conv-relu-batchnorm-layer input=cnn_res6 name=mfcc_cnn7 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  no-op-component name=cnn_res7 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7)
  conv-relu-batchnorm-layer input=cnn_res7 name=mfcc_cnn8 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  no-op-component name=cnn_res8 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8)  
  conv-relu-batchnorm-layer input=cnn_res8 name=mfcc_cnn9 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64  
  no-op-component name=cnn_res9 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8,mfcc_cnn9)  
  conv-relu-batchnorm-layer input=cnn_res9 name=mfcc_cnn10 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64  
  no-op-component name=cnn_res10 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8,mfcc_cnn9,mfcc_cnn10)  
  conv-relu-batchnorm-layer input=cnn_res10 name=mfcc_cnn11 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64  
  no-op-component name=cnn_res11 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8,mfcc_cnn9,mfcc_cnn10,mfcc_cnn11) 
  conv-relu-batchnorm-layer input=cnn_res11 name=mfcc_cnn12 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  no-op-component name=cnn_res12 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8,mfcc_cnn9,mfcc_cnn10,mfcc_cnn11,mfcc_cnn12)  
  conv-relu-batchnorm-layer input=cnn_res12 name=mfcc_cnn13 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64  
  no-op-component name=cnn_res13 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8,mfcc_cnn9,mfcc_cnn10,mfcc_cnn11,mfcc_cnn12,mfcc_cnn13)  
  conv-relu-batchnorm-layer input=cnn_res13 name=mfcc_cnn14 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64  
  no-op-component name=cnn_res14 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8,mfcc_cnn9,mfcc_cnn10,mfcc_cnn11,mfcc_cnn12,mfcc_cnn13,mfcc_cnn14)  
  conv-relu-batchnorm-layer input=cnn_res14 name=mfcc_cnn15 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64  
  no-op-component name=cnn_res15 input=Sum(fea_cnn,mfcc_cnn5,mfcc_cnn6,mfcc_cnn7,mfcc_cnn8,mfcc_cnn9,mfcc_cnn10,mfcc_cnn11,mfcc_cnn12,mfcc_cnn13,mfcc_cnn14,mfcc_cnn15)  


# the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
  # information bottleneck doesn't become a problem.
  
  tdnnf-layer name=tdnnf1 $tdnnf_first_opts dim=320 bottleneck-dim=96 time-stride=0
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3  
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3  
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3  
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3  
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3  
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=320 bottleneck-dim=96 time-stride=3  

  # final layers
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
  
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 4 ]; then
  echo =============debug4================
  steps/nnet3/chain/train.py --stage $train_stage \
    --trainer.input-model ${trained_model} \
    --use-gpu "wait" \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir "" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch ${minibatch} \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs ${num_epoch} \
    --trainer.optimization.num-jobs-initial ${init_num_jobs} \
    --trainer.optimization.num-jobs-final ${final_num_jobs} \
    --trainer.optimization.initial-effective-lrate ${init_LrRate} \
    --trainer.optimization.final-effective-lrate ${final_LrRate} \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --cuda-devices ${cuda_devices} \
    --dir $dir  || exit 1;

fi

if [ $stage -le 5 ]; then
  echo =============debug5================
    steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $langtrain $extractor $dir ${dir}_online || exit 1;
fi
exit 0;