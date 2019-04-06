#!/home/mayhew2/miniconda3/bin/python
import os,codecs
from collections import defaultdict

import random

random.seed(8122233235)

def func(lines1file, lines2file):

    with open(lines1file) as f:
        lines1 = f.readlines()

    with open(lines2file) as f:
        lines2 = f.readlines()

    both = list(zip(lines1, lines2))

    # shuffle
    random.shuffle(both)

    # sample so we have the same number of matched and unmatched.
    count = defaultdict(int)
    keep = []
    for p in both:

        a,b = p
        label = a.split("\t")[1].strip()
        
        if label == "unmatched":
            if count["neg"] < count["pos"]:        
                keep.append(p)
                count["neg"] += 1
        else:
            # always add positives
            keep.append(p)
            count["pos"] += 1

    
    # split into train and test
    total = len(keep)
    train = int(total * 0.8)
    trainlines = keep[:train]
    testlines = keep[train:]

    with open(lines1file + ".train", "w") as train1, open(lines2file + ".train", "w") as train2:
        for p in trainlines:
            a,b = p
            train1.write(a)
            train2.write(b)

    with open(lines1file + ".test", "w") as test1, open(lines2file + ".test", "w") as test2:
        for p in testlines:
            a,b = p
            test1.write(a)
            test2.write(b)

            
    print(count)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("lines1",help="")
    parser.add_argument("lines2",help="")

    args = parser.parse_args()
    
    func(args.lines1, args.lines2)
