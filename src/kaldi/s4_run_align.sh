#!/bin/bash
################################################
# Necessary files: conf/mfcc.conf 
# Produced By Xinhui Hu, 2014/06/13
###############################################
. ./cmd.sh
. ./path.sh
. ./configs.sh

DB_DIR=mfcc_pitch
ASR_MODEL_DIR=exp
EXP_DIR=$ASR_MODEL_DIR/exp_mfcc_pitch
LM_DIR=exp/lm
LEX=$LM_DIR/lang_nosp_withLM_training_dict

mfcc_config=conf/mfcc.conf
pitch_config=conf/pitch.conf
nj=16

data_name=f88_infer
data_dir=data/
output_dir=mfcc_pitch/

LAT_DATA=$DB_DIR/$data_name

cp -r $data_dir/$data_name $output_dir
steps/make_mfcc_pitch.sh \
    --mfcc-config $mfcc_config \
    --cmd "$train_cmd" \
    --nj $nj \
    --pitch-config $pitch_config \
    $output_dir/$data_name || exit 1;
utils/fix_data_dir.sh $output_dir/$data_name || exit 1;
steps/compute_cmvn_stats.sh $output_dir/$data_name || exit 1;
steps/align_fmllr.sh $output_dir/$data_name $LEX $EXP_DIR/tri5a $EXP_DIR/tri5a_$data_name || exit 1;
