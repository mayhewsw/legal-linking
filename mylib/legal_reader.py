from typing import Iterator, List
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.commands.train import *
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
import json
import random

# [
#     {
#         'text': "In 2008, the California Supreme Court held that...",
#         'meta': {
#             'doc_type': "opinion",
#             'id': 0
#             'source_url': "https://www.law.cornell.edu/supremecourt/text/12-144"
#         }
#         'matches': [["Fourteenth Amendment", "https://www.law.cornell.edu/constitution/amendmentxiv"]]
#     },
#     ...
# ]


@DatasetReader.register("legal_reader")
class LegalDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = True) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._word_splitter = SpacyWordSplitter()
        self.lazy = lazy

    def text_to_instance(self, graf_tokens: List[Token], const_tokens: List[str], label: int = None) -> Instance:
        graf_field = TextField(graf_tokens, self.token_indexers)
        const_field = TextField(const_tokens, self.token_indexers)

        fields = {"graf": graf_field, "const": const_field}

        if label is not None:
            label_field = LabelField(label, skip_indexing=True)
            fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        # load constitutional data first.
        with open(file_path + "/constitution.json") as f:
            constitution = json.load(f)

        for k in constitution.keys():
            constitution[k] = self._word_splitter.split_words(constitution[k])

        allkeys = list(constitution.keys())
        print(allkeys)

        with open(file_path + "/ussc_out_short.json") as f:
            for line in f:
                grafs = json.loads(line)
                for graf in grafs:
                    graf_text = self._word_splitter.split_words(graf["text"])

                    if len(graf["matches"]) == 0:
                        # match against random constitution para
                        key = random.choice(allkeys)
                        const_text = constitution[key]
                        yield self.text_to_instance(graf_text, const_text, 0)

                    else:
                        for match in graf["matches"]:
                            # match this text against each match.
                            match_text, key = match
                            if key not in allkeys:
                                raise Exception("Match key {} not found in all keys.".format(key))
                            const_text = constitution[key]
                            yield self.text_to_instance(graf_text, const_text, 1)



if __name__ == "__main__":
    ldr = LegalDatasetReader()
    k = 0
    for i in ldr._read("data/"):
        if i["label"].label == 1:
            print(i)
