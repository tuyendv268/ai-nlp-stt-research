export KALDI_ROOT=/opt/kaldi
export PATH=$PWD/utils:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/nnet3bin/:$KALDI_ROOT/src/onlinebin:$KALDI_ROOT/src/online2bin:$PWD:$KALDI_ROOT/src/ivectorbin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/src/cudadecoderbin:$PATH
export PATH=$KALDI_ROOT/tools/srilm/bin/i686-m64:$KALDI_ROOT/tools/srilm/bin:$PATH
export PATH=$KALDI_ROOT/tools/mitlm:$KALDI_ROOT/tools/pocolm/scripts:$KALDI_ROOT/src/cudafeatbin:$PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$KALDI_ROOT/tools/openfst/lib:/usr/local/cuda/lib64"
nj=69
njob=69
ivec_job=50 # =8/4
export LC_ALL=C
