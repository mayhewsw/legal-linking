import re
from collections import Counter

import tqdm


def tagdata(infile, outfile, destructive=False, remove=False):
    with open("mylib/rules.txt") as f:
        lines = f.readlines()
        rules = dict(map(lambda line: line.strip().split("\t"), lines))

    pat = re.compile('(' + '|'.join(rules.keys()) + ')')

    oldcounter = Counter()
    newcounter = Counter()

    with open(infile) as f, open(outfile, "w") as out:
        for line in tqdm.tqdm(f):
            text, origlabel = line.strip().split("\t")
            oldcounter[origlabel] += 1
            m = set(pat.findall(text.lower()))

            newrules = set()

            if destructive:
                newrules.add("unmatched")
            else:
                newrules.add(origlabel)

            if len(m) > 0:
                for matching_key in m:
                    if remove:
                        text = re.sub(matching_key, '', text, flags=re.IGNORECASE).strip()
                    newlabel = rules[matching_key]
                    newcounter[newlabel] += 1
                    newrules.add(newlabel)

            if len(newrules) > 1 and "unmatched" in newrules:
                newrules.remove("unmatched")

            outstring = "{}\t{}\n".format(text, ",".join(newrules))
            out.write(outstring)

    #print("tag stats before:", oldcounter)
    #print("tag stats after:", newcounter)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some rules.')
    parser.add_argument('--infile', '-i', help='File of lines to read in, probably called tmp_lines', required=True)
    parser.add_argument('--outfile', '-o', help='File to write to.', required=True)
    parser.add_argument("--destructive", "-d", help="Keep original rules or no?", default=False, action="store_true")
    parser.add_argument("--remove", "-r", help="Remove matched string spans (to avoid model overfitting)", default=False, action="store_true")


    args = parser.parse_args()
    print(args)
    tagdata(args.infile, args.outfile, args.destructive, args.remove)
