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
    # can add --limit 100 for testing.
    python mylib/json2lines.py -i data/all_json -o data/all_lines
fi

# the r means REMOVE
python mylib/tagdata.py -i data/all_lines -o data/all_lines_labeled
python mylib/tagdata.py -i data/all_lines -o data/all_lines_labeled_remove --remove


# this balances the unmatched/matched distribution (50/50) and produces a train/test split.
python mylib/preparedata.py data/all_lines_labeled data/all_lines_labeled_remove

echo "Add the constitution to the data for good measure."
python mylib/json2lines.py -d data/const
cd data
cat const all_lines_labeled.train | shuf > all.train
cat const all_lines_labeled_remove.train | shuf > all_remove.train
cat const all_lines_labeled.test | shuf > all.test
cat const all_lines_labeled_remove.test | shuf > all_remove.test

# do a little cleanup
#rm -f all_lines_labeled* const
