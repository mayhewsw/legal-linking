
for f in data/validation/all_vali*; do
    echo $f
    python mylib/tagdata.py -i $f -o rules.txt -d
    python score_rules.py -g $f -p rules.txt
done
