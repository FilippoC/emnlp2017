#!/bin/python3
import sys, collections

train_spines_path = sys.argv[1]
dev_spines_path = sys.argv[2]

def read_spines(path):
    spine_sentences = []
    with open(path) as f:
        spines = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if len(spines) > 0:
                    spine_sentences.append(spines)
                    spines = []

                continue

            line = line.split()
        
            spine = {}
            spine["id"] = int(line[0])
            spine["word"] = line[1]
            spine["pos"] = line[2]
            spine["template"] = line[3]
            spine["head"] = int(line[4])
            spine["att_position"] = int(line[5])
            spine["att_type"] = line[6]
            spines.append(spine)

        if len(spines) > 0:
            spine_sentences.append(spines)

    return spine_sentences

templates = set()
templates_per_pos = collections.defaultdict(set)
templates_in = collections.defaultdict(set)

for spines in read_spines(train_spines_path):
    for spine in spines:
        templates.add(spine["template"])
        templates_per_pos[spine["pos"]].add(spine["template"])

        if spine["pos"] == "IN":
            templates_in[spine["word"]].add(spine["template"])

print("n. templates: %i" % len(templates))
print("n. templates per pos: ")
for k, v in templates_per_pos.items():
    print("\t%s: %i" % (k, len(v)))
#print("n. templates IN: ")
#for k, v in templates_in.items():
#    print("\t%s: %i" % (k, len(v)))
print("---")

n_token = 0
n_accessibles = 0
n_accessibles_pos = 0
for spines in read_spines(dev_spines_path):
    for spine in spines:
        n_token += 1

        if spine["template"] in templates:
            n_accessibles += 1
        if spine["template"] in templates_per_pos[spine["pos"]]:
            n_accessibles_pos += 1

print("Accessible: %.2f (%i / %i)" % (100 * n_accessibles / n_token, n_accessibles, n_token))
print("Accessible with pos filtering : %.2f (%i / %i)" % (100 * n_accessibles_pos / n_token, n_accessibles_pos, n_token))

