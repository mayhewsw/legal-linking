from mylib import legal_reader
from allennlp.predictors import Predictor
from mylib.json2lines import *

from mylib import legal_model
from mylib import legal_predictor

json_conv = JsonConverter()
constitution = json_conv._read_const("data")

predictor = Predictor.from_path("tmp/model.tar.gz", "legal_predictor")
ldr = predictor._dataset_reader


def getname(pred_ind):
    return predictor._model.vocab.get_token_from_index(pred_ind, namespace="labels")


while True:
    graf = input("enter text>> ")
    if graf in ["q", "quit"]:
        break

    if len(graf.split()) < 5:
        continue
    if len(graf.strip()) == 0:
        continue

    matches = []
    out = predictor.predict_json({"graf": graf, "const": "who cares"})

    pred_ind = out["instance"]["prediction"]
    pred_name = getname(pred_ind)
    print(out)
    print(pred_name)
    if pred_name != "unmatched":
        print(constitution[pred_name])
