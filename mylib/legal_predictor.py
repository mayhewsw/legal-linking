from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('legal_predictor')
class LegalPredictor(Predictor):

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        graf = self._dataset_reader._word_splitter.split_words(json_dict['graf'])
        const = self._dataset_reader._word_splitter.split_words(json_dict['const'])
        instance = self._dataset_reader.text_to_instance(graf, const)
        return {"instance": self.predict_instance(instance)}
