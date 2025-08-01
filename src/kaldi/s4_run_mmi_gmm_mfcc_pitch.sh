#!/bin/bash
#########################################################################
# Training Kaldi GMMs using MFCC(13)+pitch(3) feature for Vietnam speech 
# Modified by Nguyen Van Huy, 2014/09/18
#
#########################################################################

. ./cmd.sh
. ./path.sh
. ./conf/common_vars.sh || exit 1;

ASR_MODEL_DIR=exp

stage=5

type=mfcc_pitch
EXP_DIR=$ASR_MODEL_DIR/exp_${type}


# training acoustic models
DB_DIR=mfcc_pitch
DB_NAME=f88_train
LM_DIR=exp/lm

TRAIN_mono=$DB_DIR/TRAIN_mono
# TRAIN_mono=$DB_DIR/$DB_NAME
TRAIN_tri1=$DB_DIR/$DB_NAME
TRAIN_tri2=$DB_DIR/$DB_NAME
TRAIN_tri3=$DB_DIR/$DB_NAME
TRAIN_FULL=$DB_DIR/$DB_NAME
LAT_DATA=$DB_DIR/$DB_NAME

LEX=$LM_DIR/lang_nosp_withLM_training_dict

njob=12

train_opt="--nj $njob --boost-silence $boost_sil --cmd $train_cmd"
train_opt1="--boost-silence $boost_sil --cmd $train_cmd"

if [ ${stage} -le 0 ]; then
  echo --- mono --- 
  echo "Step 06: Starting monophone training in $EXP_DIR/mono" `date`
  echo ---------------------------------------------------------------------
      ./utils/subset_data_dir.sh $TRAIN_FULL 696000 $TRAIN_mono || exit 1;
      steps/train_mono.sh $train_opt $TRAIN_mono $LEX $EXP_DIR/mono || exit 1;    
      utils/mkgraph.sh ${LEX} $EXP_DIR/mono $EXP_DIR/mono/graph
fi

if [ ${stage} -le 1 ]; then
  echo ---------------------------------------------------------------------- 
  echo "Step 06: triphone training in $EXP_DIR/tri1 on" `date`
  echo ---------------------------------------------------------------------
      # ./utils/subset_data_dir.sh $TRAIN_FULL 2000000 $TRAIN_tri1 || exit 1;
      steps/align_si.sh $train_opt $TRAIN_tri1 $LEX $EXP_DIR/mono $EXP_DIR/mono_ali || exit 1;
      echo "numGaussTri1=" $numGaussTri1 "numLeavesTri1=" $numLeavesTri1
      steps/train_deltas.sh $train_opt1 $numLeavesTri1 $numGaussTri1 $TRAIN_tri1 $LEX $EXP_DIR/mono_ali $EXP_DIR/tri1 || exit 1;

      utils/mkgraph.sh ${LEX} $EXP_DIR/tri1 $EXP_DIR/tri1/graph
fi

if [ ${stage} -le 2 ]; then
  echo ---------------------------------------------------------------------
  echo "Step 06: triphone training in $EXP_DIR/tri2 on" `date`
  echo ---------------------------------------------------------------------
    # ./utils/subset_data_dir.sh $TRAIN_FULL 4000000 $TRAIN_tri2 || exit 1;
    steps/align_si.sh $train_opt $TRAIN_tri2 $LEX $EXP_DIR/tri1 $EXP_DIR/tri1_ali || exit 1;
    steps/train_deltas.sh $train_opt1 $numLeavesTri2 $numGaussTri2 $TRAIN_tri2 $LEX $EXP_DIR/tri1_ali $EXP_DIR/tri2 || exit 1;
    utils/mkgraph.sh ${LEX} $EXP_DIR/tri2 $EXP_DIR/tri2/graph
fi

if [ ${stage} -le 3 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting triphone training in $EXP_DIR/tri3 on" `date`
  echo ---------------------------------------------------------------------
    # ./utils/subset_data_dir.sh $TRAIN_FULL 8000000 $TRAIN_tri3 || exit 1;
    steps/align_si.sh $train_opt $TRAIN_tri3 $LEX $EXP_DIR/tri2 $EXP_DIR/tri2_ali || exit 1;
    steps/train_deltas.sh $train_opt1 $numLeavesTri3 $numGaussTri3 $TRAIN_tri3 $LEX $EXP_DIR/tri2_ali $EXP_DIR/tri3 || exit 1;
    utils/mkgraph.sh ${LEX} $EXP_DIR/tri3 $EXP_DIR/tri3/graph
fi

if [ ${stage} -le 4 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (lda_mllt) triphone training in $EXP_DIR/tri4 on" `date`
  echo ---------------------------------------------------------------------

    steps/align_si.sh $train_opt $TRAIN_FULL $LEX $EXP_DIR/tri3 $EXP_DIR/tri3_ali || exit 1; 
    steps/train_lda_mllt.sh $train_opt1 $numLeavesMLLT $numGaussMLLT $TRAIN_FULL $LEX $EXP_DIR/tri3_ali $EXP_DIR/tri4 || exit 1;
    utils/mkgraph.sh ${LEX} $EXP_DIR/tri4 $EXP_DIR/tri4/graph
fi

if [ ${stage} -le 5 ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (SAT) triphone training in $EXP_DIR/tri5a on" `date`
  echo ---------------------------------------------------------------------

    # steps/align_si.sh $train_opt $TRAIN_FULL $LEX $EXP_DIR/tri4 $EXP_DIR/tri4_ali || exit 1;
    steps/train_sat.sh $train_opt1 $numLeavesSATa $numGaussSATa $TRAIN_FULL $LEX $EXP_DIR/tri4_ali $EXP_DIR/tri5a || exit 1;
    utils/mkgraph.sh ${LEX} $EXP_DIR/tri5a $EXP_DIR/tri5a/graph
fi

# if [ ${stage} -le 6 ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting align and make lat for $EXP_DIR/tri5a on" `date`
#   echo ---------------------------------------------------------------------

#     steps/align_fmllr.sh $train_opt $TRAIN_FULL $LEX $EXP_DIR/tri5a $EXP_DIR/tri5a_ali || exit 1;
#     steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" $LAT_DATA \
#      $LEX $EXP_DIR/tri5a $EXP_DIR/tri5a_lat
# fi

echo "Done!!!!!!"