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
        pred_name = self._model.vocab.get_token_from_index(pred_ind, namespace="labels")
        const_text = "[None]"
        const_link = "#"
        if pred_name != "unmatched":
            const_text = self.constitution[pred_name]
            const_link = self.links[pred_name]
        return {"instance": result, "const_text": const_text, "const_link": const_link}
