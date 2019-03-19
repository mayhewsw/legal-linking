import sys
import json
import tqdm
from collections import Counter

fname = sys.argv[1]

total = 0
totalnonempty = 0
matches = 0
parawithmatches = 0
uniquematches = 0
uniqueparawithmatches = 0
seen = Counter()

with open(fname) as f:
    for line in tqdm.tqdm(f):
        js = json.loads(line)
        total += len(js)
        for graf in js:

            if graf["text"] not in seen:
                uniquematches += len(graf["matches"])
                if len(graf["matches"]) > 0:
                    uniqueparawithmatches += 1

            seen[graf["text"]] += 1

            if len(graf["text"].strip()) > 0:
                totalnonempty += 1
            if len(graf["matches"]) > 0:
                parawithmatches += 1
            matches += len(graf["matches"])

print("total paragraphs", total)
print("total unique", len(seen))
print("total nonempty para", totalnonempty)
print("total num matches", matches)
print("num paragraphs with matches", parawithmatches)
print("total unique matches", uniquematches)
print("total unique paras with matches", uniqueparawithmatches)


print("========== 10 most common lines ===========")
for k in seen.most_common(10):
    print(k)