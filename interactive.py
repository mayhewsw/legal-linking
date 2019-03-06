from mylib import legal_reader
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from json2lines import *

from mylib import legal_model
from mylib import legal_predictor

json_conv = JsonConverter()
constitution = json_conv._read_const("data")

#for k in constitution:
#    print(constitution[k])

ldr = legal_reader.LegalDatasetReader()

archive = load_archive('tmp/model.tar.gz')
model = archive.model
model.eval()

predictor = Predictor.by_name("legal_predictor")(model, ldr)


while True:
    graf = input("enter text>> ")
    if graf in ["q", "quit"]:
        break

    if len(graf.strip()) == 0:
        continue

    num_matched=0
    for k in constitution:
        const = constitution[k]
        out = predictor.predict_json({"graf": graf, "const": const})
        if out["instance"]["prediction"] == 1:
            print(k, out)
            print(constitution[k])
            num_matched += 1
    print(num_matched)
