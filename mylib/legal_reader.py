from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.commands.train import *
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
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

    def _read_const(self, file_dir):
        """
        This reads the constitution.json file and returns a dictionary.
        :param file_dir:
        :return:
        """

        # load constitutional data first.
        with open(file_dir + "/constitution.json") as f:
            infile = json.load(f)

        # there are some duplicate keys in the data.
        texts = set()

        constitution = {}

        for k in infile.keys():
            txt = infile[k]["text"].strip()
            if len(txt) > 0:
                if txt not in texts:
                    # TODO: consider also passing metadata in
                    constitution[k] = self._word_splitter.split_words(txt)
                    texts.add(txt)

        return constitution

    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        file_path: must be in the same folder as constitution.json
        """
        file_dir = dirname(file_path)

        # dict: {key : [words, of, constitution], ...}
        constitution = self._read_const(file_dir)
        allkeys = list(constitution.keys())

        counts = {"pos": 0, "neg": 0}

        with open(file_path) as f:
            lines = f.readlines()

        print("============== ONLY THE FIRST 1000 EXAMPLES========================")
        for line in lines:
            grafs = json.loads(line)
            if sum(counts.values()) > 10000:
                break
            for graf in grafs:
                # doesn't make sense to have empty text?
                if len(graf["text"].strip()) == 0:
                    graf["text"] = "empty"
                    continue

                # List[str]
                graf_text = self._word_splitter.split_words(graf["text"])

                # A set of all keys which are definitely negative
                # (according to the supervision we have)
                nonmatchingkeys = set(allkeys)

                # if there are no matches, this won't run. NP
                for match in graf["matches"]:
                    match_text, match_link, grafkey = match
                    const_text = constitution[grafkey]
                    counts["pos"] += 1
                    if grafkey in nonmatchingkeys:
                        nonmatchingkeys.remove(grafkey)
                    yield self.text_to_instance(graf_text, const_text, 1)

                nonmatchingkeys = list(nonmatchingkeys)

                # allow up to 3x negatives.
                while counts["neg"] < 3*counts["pos"]:
                    # match against random constitution para
                    key = random.choice(nonmatchingkeys)
                    const_text = constitution[key]
                    counts["neg"] += 1
                    yield self.text_to_instance(graf_text, const_text, 0)

        print(counts)



if __name__ == "__main__":
    ldr = LegalDatasetReader()
    k = 0
    for i in ldr._read("data/ussc_out_dev_0.json"):
        print(i)
        if i["label"].label == 1:
            print(i)
