import re
from collections import Counter

import tqdm


def tagdata(infile, outfile):
    with open("mylib/rules.txt") as f:
        lines = f.readlines()
        rules = dict(map(lambda line: line.strip().split("\t"), lines))

    print("rules", rules)

    pat = re.compile('(' + '|'.join(rules.keys()) + ')')

    oldcounter = Counter()
    newcounter = Counter()

    seen = set()

    with open(infile) as f, open(outfile, "w") as out:
        for line in tqdm.tqdm(f):
            text, origlabel = line.strip().split("\t")
            oldcounter[origlabel] += 1
            m = set(pat.findall(text.lower()))
            if len(m) > 0:
                newrules = set()
                if origlabel != "unmatched":
                    newrules.add(origlabel)

                for matching_key in m:
                    newlabel = rules[matching_key]
                    newcounter[newlabel] += 1
                    newrules.add(newlabel)

                outstring = "{}\t{}\n".format(text, ",".join(newrules))
                out.write(outstring)
            else:
                out.write(line)
                newcounter[origlabel] += 1
    print("tag stats before:", oldcounter)
    print("tag stats after:", newcounter)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some rules.')
    parser.add_argument('--infile', '-i', help='File of lines to read in, probably called tmp_lines')
    parser.add_argument('--outfile', '-o', help='File to write to.')

    args = parser.parse_args()
    tagdata(args.infile, args.outfile)
