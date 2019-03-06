with open("test_1col.sents") as f:
    sents = f.read().split("\n")

def write(cons,outname):
    with open(outname, "w") as out:
        for c in cons:
            if len(c.split()) < 5:
                continue
            for k in cons:
                if c == k:
                    out.write("{}\t{}\t{}\n".format(c, k, 1))
                else:
                    out.write("{}\t{}\t{}\n".format(c, k, 0))


write(sents[:10], "tmp_train")
write(sents[10:15], "tmp_dev")
write(sents[15:20], "tmp_test")
