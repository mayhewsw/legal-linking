from mylib import legal_reader
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

from mylib import legal_model
from mylib import legal_predictor

ldr = legal_reader.LegalDatasetReader()
constitution = ldr._read_const("data")
print(constitution.keys())

#print(constitution["0"])

print("loading a default dataset reader...")

archive = load_archive('tmp/model.tar.gz')
model = archive.model
model.eval()

predictor = Predictor.by_name("legal_predictor")(model, ldr)

graf = "Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof."


for k in constitution:
    const = constitution[k]
    out = predictor.predict_json({"graf": graf, "const": const})
    if out["instance"]["prediction"] == 1:
        print(k, out)
        print(constitution[k])
