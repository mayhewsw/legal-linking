#!/usr/bin/env bash

set -e

TOTAL=$1
if [[ "$#" -ne 1 ]]; then
  # set TOTAL to a default value of HUGE
  TOTAL=1000000000000000000
fi


# thanks: https://stackoverflow.com/questions/5914513/shuffling-lines-of-a-file-with-a-fixed-seed
get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
            </dev/zero 2>/dev/null
}

# combine all input files.
rm -f all_json
for f in ussc_out_full*;
do
    cat $f >> all_json
done

echo "Total num json lines: "
wc -l all_json

cd ..
if [[ ! -f data/all_lines ]]; then
    echo "File not found!"
    echo "Converting to lines"
    python mylib/json2lines.py -i data/all_json -o data/all_lines
fi

python mylib/tagdata.py -i data/all_lines -o data/all_lines_labeled


cd data
shuf --random-source=<(get_seeded_random 42) all_lines_labeled | head -n $TOTAL > tmp

# sample the number of unmatched.
grep "unmatched$" tmp > unmatched
grep -v "unmatched$" tmp > matched

NUMMATCHED=$(wc -l matched | awk '{print $1}')

head -n $(($NUMMATCHED * 2)) unmatched > newunmatched
cat matched newunmatched | shuf > tmp
rm matched unmatched newunmatched

grep -c "unmatched$" tmp
grep -vc "unmatched$" tmp

# If $TOTAL is not given as argument, then it gets a default value.
TOTAL_TMP=$(wc -l tmp | cut -f 1 -d' ')
TRAIN=$(($TOTAL_TMP * 4 / 5))
DT=$(($T - $TRAIN))

head -n $TRAIN tmp > train
tail -n $DT tmp > devtest
head -n $((DT / 2)) devtest > dev
tail -n $((DT / 2)) devtest > test

echo "Add the constitution to the data for good measure."
cd ..
python mylib/json2lines.py -d data/const
cat data/const data/train | awk 'length($0)<1000' | shuf > data/train2
cat data/const data/dev | awk 'length($0)<1000' | shuf > data/dev2
cat data/const data/test | awk 'length($0)<1000' | shuf > data/test2

cd data

mv train2 train
mv dev2 dev
mv test2 test

rm const
rm tmp
rm devtest

wc -l train
wc -l dev
wc -l test
