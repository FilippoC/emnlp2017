#!/usr/bin/python3

import subprocess, tempfile, sys, copy, sys, os

gold_path = sys.argv[1]
pred_path = sys.argv[2]
pos_index = 4;
if len(sys.argv) > 3 and sys.argv[3] == "cpos":
    pos_index = 3;

def sentence_generator(path):
    with open(path) as handler:
        yield_data = []
        for line in handler:
            line = line.strip()
            if len(line) == 0:
                if len(yield_data) > 0:
                    yield yield_data
                    yield_data = []
            else:
                line = line.split()
                yield_data.append(line)

        if len(yield_data) > 0:
            yield yield_data

total = 0
correct = 0

for gold, pred in zip(sentence_generator(gold_path), sentence_generator(pred_path)):
    for w1, w2 in zip(gold, pred):
        total += 1
        correct += 1 if w1[pos_index] == w2[pos_index] else 0

print("Correct pos: %.2f (%i / %i)"%(100 * correct / total, correct, total))
