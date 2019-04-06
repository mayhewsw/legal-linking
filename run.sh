
set -e

#export TRAIN_DATA=data/all.train
#export TEST_DATA=data/all.test

#rm -rf /save/mayhew2/legal-bert
#allennlp train legal_gpu.json --include-package mylib -s /save/mayhew2/legal-bert/


export TRAIN_DATA=data/all_remove.train
export TEST_DATA=data/all_remove.test

rm -rf /save/mayhew2/legal-bert-remove
allennlp train legal_gpu.json --include-package mylib -s /save/mayhew2/legal-bert-remove/
