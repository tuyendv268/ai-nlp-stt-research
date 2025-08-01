#!/bin/bash
. ./path.sh
. ./cmd.sh
. ./configs.sh

#
ali_in_dir=/data/asr-research/src/kaldi/exp/exp_mfcc_pitch/tri5a_f88_infer
ali_out_dir=/data/asr-research/src/kaldi/exp/exp_mfcc_pitch/tri5a_f88_infer_out
mkdir $ali_out_dir

echo "Process alignments"
for data in $ali_in_dir/ali.*.gz
do
    echo $data
    fbname=$(basename "$data" .gz)
    echo "Show alignments $fbname"
    
    show-alignments $ali_in_dir/phones.txt \
        $ali_in_dir/final.mdl \
        'ark:gunzip -c '${data}'|' > ${ali_out_dir}/${fbname}.txt

    ali-to-phones --per-frame \
        --ctm-output \
        $ali_in_dir/final.mdl \
        "ark:gunzip -c "${data}"|" -> ${out_ctm_dir}/${fbname}.ctm
done

echo "Done!!!" 
