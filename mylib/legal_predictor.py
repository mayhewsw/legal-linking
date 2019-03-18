from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides
import sys

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from mylib.json2lines import JsonConverter


@Predictor.register('legal_predictor')
class LegalPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader):
        super().__init__(model, dataset_reader)
        json_conv = JsonConverter()
        self.spacy = SpacyWordSplitter()
        self.constitution, self.links = json_conv._read_const("data")

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        graf = self.spacy.split_words(json_dict['graf'])
        instance = self._dataset_reader.text_to_instance(graf)
        result = self.predict_instance(instance)
        pred_ind = result["prediction"]
        predictions = set()
        for i,pred in enumerate(pred_ind):
            if pred != 0:
                pred_name = self._model.vocab.get_token_from_index(i, namespace="labels")
                predictions.add(pred_name)

        probs = []
        for i,prob in enumerate(result["class_probabilities"]):
            class_name = self._model.vocab.get_token_from_index(i, namespace="labels")
            probs.append((class_name, prob))

        # we want to sort according to the probability of a positive prediction.
        probs = sorted(probs, key=lambda p: p[1][1], reverse=True)
        result["class_probabilities"] = probs
        
        predictions = list(predictions)
        if len(predictions) > 0:
            pred_name = predictions[0]
        else:
            pred_name = "unmatched"
            
        const_text = "unmatched"
        const_link = "#"
        if pred_name != "unmatched":
            const_text = self.constitution[pred_name]
            const_link = self.links[pred_name]
        return {"instance": result, "const_text": predictions, "const_link": const_link, "graf" : graf}

    def dump_line(self, outputs: JsonDict):
        print(outputs)
        return outputs["graf"] + "\t" + outputs["const_text"] + "\n"

