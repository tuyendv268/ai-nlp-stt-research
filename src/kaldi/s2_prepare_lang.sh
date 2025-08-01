#!/bin/bash
. ./path.sh
. ./cmd.sh
. ./configs.sh
stage=1
data_dir=data
lm_dir=exp/lm
train_lm=$lm_dir/lm_train.arpa.gz

train_dict_name=dict

#############################
if [ $stage -le 1 ]; then	
	cp ${data_dir}/lexicon lexicon
	# echo "<noise> +noise+" >> lexicon
	echo "ah +ah+" >> lexicon
	echo "uhm +uhm+" >> lexicon
	echo "oh +oh+" >> lexicon
	echo "wow +wow+" >> lexicon
	echo "<unk> +unk+" >> lexicon
	cat lexicon | sort | uniq > tmp.txt
	mv tmp.txt lexicon
	dict=lexicon
	echo "Step 03.01 Preparing Lexicon for training"
	local/fptasr_prepare_dict.sh ${dict} ${lm_dir}/local ${train_dict_name} || exit 1;
	cat $lm_dir/local/${train_dict_name}/nonsilence_phones.txt \
		| grep -w -v "+ah+" \
		| grep -w -v "+uhm+" \
		| grep -w -v "+oh+" \
		| grep -w -v "+wow+"  > tmp.txt
	mv tmp.txt $lm_dir/local/${train_dict_name}/nonsilence_phones.txt
	echo "+ah+" >> $lm_dir/local/${train_dict_name}/silence_phones.txt
	echo "+uhm+" >> $lm_dir/local/${train_dict_name}/silence_phones.txt
	echo "+oh+" >> $lm_dir/local/${train_dict_name}/silence_phones.txt
	echo "+wow+" >> $lm_dir/local/${train_dict_name}/silence_phones.txt
	# echo "+noise+" >> $lm_dir/local/${train_dict_name}/silence_phones.txt
fi

################# Prepare Lang for traning
if [ $stage -le 3 ]; then
	langout=$lm_dir/lang_${train_dict_name}
	langwithlmout=$lm_dir/lang_nosp_withLM_training_${train_dict_name}
	rm -rf ${langout}
	echo "Step 03.03: Preparing Lang for traning using Lexicon"
	rm -rf ${langout}
	utils/prepare_lang.sh $lm_dir/local/${train_dict_name} "<unk>" $lm_dir/local/lang_${train_dict_name} ${langout} || exit 1 ;
	rm -rf ${langwithlmout}
	echo "Step 03.04: Preparing Lang for traning using Language model"
	#local/arpa2G.sh $train_lm $lm_dir/lang_${outdict} $lm_dir/lang_${outdict} || exit 1;
	srilm_opts="-subset -prune-lowprobs -order 3"
	utils/format_lm_sri.sh --srilm-opts "$srilm_opts" ${langout} $train_lm $lm_dir/local/${train_dict_name}/lexicon.txt ${langwithlmout}
fi
