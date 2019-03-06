from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.commands.train import *
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, JustSpacesWordSplitter
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
import json
import random
from tqdm import tqdm

from os.path import dirname

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
                 lazy: bool = False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._word_splitter = JustSpacesWordSplitter()
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
        """
        file_path: must be in the same folder as constitution.json
        """

        counts = {"pos": 0, "neg": 0}

        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            graf_str, const_str, label_str = line.split("\t")
            if int(label_str) == 1:
                counts["pos"] += 1
            else:
                counts["neg"] += 1

            yield self.text_to_instance(self._word_splitter.split_words(graf_str),
                                        self._word_splitter.split_words(const_str),
                                        int(label_str))

        print(counts)



if __name__ == "__main__":
    ldr = LegalDatasetReader()
    k = 0
    for i in tqdm(ldr._read("outfile")):
        k += 1

