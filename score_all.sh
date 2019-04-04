
for f in data/validation/*.line; do
    echo $f
    python mylib/tagdata.py -i $f -o tmp.txt -d
    python score_rules.py -g $f -p tmp.txt
    rm tmp.txt
done
