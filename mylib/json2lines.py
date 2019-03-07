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
import spacy

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


class JsonConverter(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._word_splitter = SpacyWordSplitter()

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
                    constitution[k] = " ".join(map(str, self._word_splitter.split_words(txt)))
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

        jslines = [json.loads(line) for line in lines]

        for grafs in tqdm(jslines):
            for graf in grafs:
                # doesn't make sense to have empty text?
                if len(graf["text"].strip()) == 0:
                    graf["text"] = "empty"
                    continue

                # str
                graf_text = " ".join(map(str, self._word_splitter.split_words(graf["text"])))

                # A set of all keys which are definitely negative
                # (according to the supervision we have)

                # if there are no matches, this won't run. NP
                if len(graf["matches"]) > 0:
                    for match in graf["matches"]:
                        match_text, match_link, grafkey = match
                        const_text = constitution[grafkey]
                        counts["pos"] += 1
                        #if grafkey in nonmatchingkeys:
                        #        nonmatchingkeys.remove(grafkey)
                        yield (graf_text, const_text, grafkey)
                else:
                    if counts["neg"] < 5*counts["pos"]:
                        counts["neg"] += 1
                        yield(graf_text, "@@empty@@", "unmatched")

        print(counts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--infile', '-i', help='File of json to read in, probably called train/test/dev')
    parser.add_argument('--outfile', '-o', help='File to write to.')
    parser.add_argument('--dumpconst', '-d', help='Dump the constitution as training data to this file.')

    args = parser.parse_args()

    ldr = JsonConverter()

    if args.dumpconst:
        nlp = spacy.load('en')
        print("writing constitution lines to", args.dumpconst)
        const = ldr._read_const("data")
        with open(args.dumpconst, "w") as out:
            for k in const:
                seen = set()
                outline = "{}\t{}\t{}\n".format(const[k], "@@empty@@", k)
                out.write(outline)
                seen.add(outline)

                # split also by sentences.
                doc = nlp(const[k])
                for sent in doc.sents:
                    outline = "{}\t{}\t{}\n".format(sent, "@@empty@@", k)
                    if outline not in seen:
                        out.write(outline)
                        seen.add(outline)

    else:
        seen = set()
        with open(args.outfile, "w") as out:
            for trip in ldr._read(args.infile):
                outline = "\t".join(trip) + "\n"
                if outline not in seen:
                    out.write(outline)
                    seen.add(outline)
