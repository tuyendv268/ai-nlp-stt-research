#!/bin/bash
################################################
# Necessary files: conf/mfcc.conf 
# Produced By Xinhui Hu, 2014/06/13
###############################################
. ./cmd.sh
. ./path.sh
. ./configs.sh

mfcc_config=conf/mfcc.conf
pitch_config=conf/pitch.conf
nj=8
echo "Step 05: Computing MFCC-Pitch feature using config from $mfcc_config $pitch_config"

data_dir=data
output_dir=mfcc_pitch
for part in f88_train;
do	
	cp -rf $data_dir/$part $output_dir
	steps/make_mfcc_pitch.sh \
		--mfcc-config $mfcc_config \
		--cmd "$train_cmd" \
		--nj $nj \
		--pitch-config $pitch_config \
		$output_dir/$part || exit 1;
    	utils/fix_data_dir.sh $output_dir/$part || exit 1;
    	steps/compute_cmvn_stats.sh $output_dir/$part || exit 1;
done
echo "Step 05: Done!!!"	
