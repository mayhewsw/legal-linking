from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.commands.train import *
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, JustSpacesWordSplitter
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
import json
import random

from overrides import overrides
from tqdm import tqdm


@DatasetReader.register("legal_reader")
class LegalDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._word_splitter = JustSpacesWordSplitter()
        self.lazy = lazy

    @overrides
    def text_to_instance(self, graf_tokens: List[Token], const_tokens: List[str], label: str = None) -> Instance:
        graf_field = TextField(graf_tokens, self.token_indexers)
        const_field = TextField(const_tokens, self.token_indexers)

        metadata = MetadataField(({"graf_words": graf_tokens, "const_tokens": const_tokens}))

        fields = {"graf": graf_field, "const": const_field, "metadata": metadata}

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
                                        label_str)

        print(counts)


if __name__ == "__main__":
    ldr = LegalDatasetReader()
    k = 0
    for i in tqdm(ldr._read("outfile")):
        k += 1
