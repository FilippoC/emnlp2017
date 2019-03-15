#!/bin/python3

import discodop.treebank
import sys, os

dptb_path = sys.argv[1]
ptb_size_path = sys.argv[2]
output_path = sys.argv[3]

part_sizes = {}
with open(ptb_size_path) as f:
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue

        line = line.split()
        part_sizes[int(line[0])] = int(line[1])


def compute_bound(part_sizes, p_begin, p_end):
    begin = 0
    end = 0

    for p, s in part_sizes.items():
        if p < p_begin:
            begin += s
        if p <= p_end:
            end += s

    return begin, end

begin_train, end_train = compute_bound(part_sizes, 2, 21)
begin_dev, end_dev = compute_bound(part_sizes, 22, 22)
begin_test, end_test = compute_bound(part_sizes, 23, 23)

reader = discodop.treebank.NegraCorpusReader(dptb_path)
trees = reader.trees()
sentences = reader.sents()

def export(trees, sentences, begin, end, path):
    with open(path, "w") as f:
        for i, k in enumerate(trees):
            if i >= end:
                break
            if i < begin:
                continue

            f.write(discodop.treebank.EXPORTHEADER)
            f.write("\n")

            f.write(discodop.treebank.writetree(trees[k], sentences[k], k, "export"))
            f.write("\n")

export(trees, sentences, begin_train, end_train, os.path.join(output_path, "cptb.train"))
export(trees, sentences, begin_dev, end_dev, os.path.join(output_path, "cptb.dev"))
export(trees, sentences, begin_test, end_test, os.path.join(output_path, "cptb.test"))
