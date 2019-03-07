#!/usr/bin/env bash

# thanks: https://stackoverflow.com/questions/5914513/shuffling-lines-of-a-file-with-a-fixed-seed
get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
            </dev/zero 2>/dev/null
}

# combine all input files.
touch tmpall
for f in ussc_out_full*;
do
    cat $f >> tmpall
done

# this maintains a standard order.
shuf --random-source=<(get_seeded_random 42) tmpall > tmp

# tmpall has about 11K items.
head -n 800 tmp > train
tail -n 200 tmp > devtest
head -n 100 devtest > dev
tail -n 100 devtest > test
rm tmp
rm tmpall
rm devtest

wc -l train
wc -l dev
wc -l test

echo "Converting to faster format"
cd ..
python mylib/json2lines.py -i data/train -o data/train_lines
python mylib/json2lines.py -i data/dev -o data/dev_lines
python mylib/json2lines.py -i data/test -o data/test_lines

echo "Mix 3 copies of constitution with training data."
python mylib/json2lines.py -d data/const
cat data/const data/const data/const data/train_lines > tmpall
shuf --random-source=<(get_seeded_random 42) tmpall > data/train_lines

cat data/const data/dev_lines > tmpall
shuf --random-source=<(get_seeded_random 42) tmpall > data/dev_lines

cat data/const data/test_lines > tmpall
shuf --random-source=<(get_seeded_random 42) tmpall > data/test_lines

rm tmpall