#!/bin/python3

import glob, sys, os

ptb_path = sys.argv[1]

def count_sentence(path):
    n = 0
    with open(path) as f:
        p = 0
        for line in f:
            for c in line:
                if c == "(":
                    if p == 0:
                        n += 1
                    p += 1
                elif c == ")":
                    p -= 1
                    #print(p)
    return n

for part_path in glob.glob(os.path.join(ptb_path, "[0-9][0-9]/")):
    part = part_path[-3:-1]
    
    size = 0
    for path in glob.glob(part_path + "wsj_" + "[0-9]" * 4 + ".mrg"):
        size += count_sentence(path)
        
    print("%s\t%i"%(part, size))
