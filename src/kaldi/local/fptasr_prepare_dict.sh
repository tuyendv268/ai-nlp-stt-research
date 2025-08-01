#!/bin/bash

# Copyright 2012 Vassil Panayotov
# Apache 2.0

. path.sh || exit 1

dict_in=$1 #lexicon
# locdata=data/local ---- old code
locdata=$2 #lm_dir
locdict=$locdata/$3
echo "***************************************** $locdict"
mkdir -p $locdict
echo "mkdir $locdict done"

#echo '<unk> unk' >> $dict_in
cat $dict_in |  sort -u  > $locdict/lexicon.txt

echo "--- Prepare phone lists ..."
echo SIL > $locdict/silence_phones.txt
echo SIL > $locdict/optional_silence.txt
grep -v -w sil $locdict/lexicon.txt | \
  awk '{for(n=2;n<=NF;n++) { p[$n]=1; }} END{for(x in p) {print x}}' |\
  sort > $locdict/nonsilence_phones.txt

echo "--- Adding SIL to the lexicon ..."
echo -e "!SIL\tSIL" >> $locdict/lexicon.txt

# Some downstream scripts expect this file exists, even if empty
touch $locdict/extra_questions.txt

echo "*** Dictionary preparation finished!"
