import sys
from collections import Counter

fname = sys.argv[1]

with open(fname) as f:
    lines = f.readlines()

label_counter = Counter()

link_counter = Counter()

for line in lines:
    sline = line.strip().split("\t")
    label = int(sline[-1])
    label_counter[label] += 1

    if label == 1:
        link_counter[sline[1]] += 1

print(label_counter)

for k in link_counter:
    print(link_counter[k])
    print(k)