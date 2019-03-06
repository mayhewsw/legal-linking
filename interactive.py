from mylib import legal_reader
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from json2lines import *

from mylib import legal_model
from mylib import legal_predictor

json_conv = JsonConverter()
constitution = json_conv._read_const("data")
print(constitution.keys())

ldr = legal_reader.LegalDatasetReader()

archive = load_archive('tmp/model.tar.gz')
model = archive.model
model.eval()

predictor = Predictor.by_name("legal_predictor")(model, ldr)

#graf = "Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press;"
graf = "i'm stephen mayhew and i'm a little crazy"
#amendment_keys = {'2', '7', '9', '10', '11', '14', '19', '23', '24', '27', '28', '32', '33', '34', '37', '42', '44',
                  #'46', '49', '50', '54', '57', '59', '61', '62', '64', '65'}

for k in constitution:
    #if k not in amendment_keys:
     #   continue
    const = constitution[k]
    out = predictor.predict_json({"graf": graf, "const": const})
    if out["instance"]["prediction"] == 1:
        print(k, out)
        print(constitution[k])
        #print(" ".join(map(str, constitution[k][:30])))
