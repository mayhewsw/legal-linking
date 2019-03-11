from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, LabelField, MetadataField, MultiLabelField
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
    def text_to_instance(self, graf_tokens: List[Token], labels: List[str] = None) -> Instance:
        graf_field = TextField(graf_tokens, self.token_indexers)

        metadata = MetadataField(({"graf_words": graf_tokens}))

        fields = {"graf": graf_field, "metadata": metadata}

        if labels is not None:
            label_field = MultiLabelField(labels)
            fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        This is a file that has been created by json2lines.
        :param file_path:
        :return:
        """

        counts = {"pos": 0, "neg": 0}

        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            graf_str, label_str = line.strip().split("\t")
            if "unmatched" == label_str:
                counts["neg"] += 1
            else:
                counts["pos"] += 1

            yield self.text_to_instance(self._word_splitter.split_words(graf_str),
                                        label_str.split(","))

        print(counts)


if __name__ == "__main__":
    ldr = LegalDatasetReader()
    k = 0
    for i in tqdm(ldr._read("outfile")):
        k += 1
