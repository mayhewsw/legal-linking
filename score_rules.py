from allennlp.data import Vocabulary

from mylib.legal_reader import *
from mylib.vectorf1 import *

# there are two files.
# open each file


def score(gold, pred):

    ldr = LegalDatasetReader()


    gold_insts = []
    all_insts = []
    for gi in ldr._read(gold):
        gold_insts.append(gi)
        all_insts.append(gi)

    pred_insts = []
    for pi in ldr._read(pred):
        pred_insts.append(pi)
        all_insts.append(pi)

    vocab = Vocabulary.from_instances(all_insts)

    for gi,pi in zip(gold_insts, pred_insts):
        gi["label"].index(vocab)
        pi["label"].index(vocab)

    f1metric = VectorF1(unmatched_index=vocab.get_token_index("unmatched", "labels"))

    for gi,pi in zip(gold_insts, pred_insts):
        gtens = gi["label"].as_tensor(None)
        ptens = pi["label"].as_tensor(None)

        if not set(gi["label"].labels) == set(pi["label"].labels):
            print(" ".join(map(str, gi["graf"].tokens)))
            print(gi["label"].labels, pi["label"].labels)

        # predicted first, then gold!!!
        f1metric(ptens, gtens)

    for n,m in zip(["P", "R", "F1"], f1metric.get_metric()):
        print("{}: {}".format(n,m))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some rules.')
    parser.add_argument('--gold', '-g', help='Gold')
    parser.add_argument('--pred', '-p', help='Predictions')

    args = parser.parse_args()
    score(args.gold, args.pred)
