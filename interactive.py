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

    matches = []
    out = predictor.predict_json({"graf": graf, "const": "who cares"})
    print(out)
