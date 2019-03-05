shuf ussc_out_full_0.json > tmp
head -n 1000 tmp > train
tail -n 500 tmp > devtest
head -n 250 devtest > dev
tail -n 250 devtest > test
rm tmp
rm devtest
