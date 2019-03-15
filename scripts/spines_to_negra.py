#!/bin/python3

import discodop.treebank
import collections
import discodop.heads
import itertools
import sys

class Node():
    def __init__(self, label, projection):
        self.label = label
        self.projection = projection
        self.children = []
        
    def add_child(self, node, att_type):
        self.children.append((node, att_type))
        
    def left_children(self):
        return [c for c in self.children if c[0].projection < self.projection]
    
    def right_children(self):
        return [c for c in self.children if c[0].projection > self.projection]
    
    def projection_child(self):
        return next(c for c in self.children if c[0].projection == self.projection)
    
    def to_discodop_tree(self):
        if len(self.children) == 0:
            # sentences are 0-indexed in discodop
            return discodop.treebank.ParentedTree(self.label, [self.projection - 1])
        else:
            return discodop.treebank.ParentedTree(
                self.label,
                [c[0].to_discodop_tree() for c in self.children]
            )
                
    def __str__(self):
        if len(self.children) == 0:
            return self.label + "-" + str(self.projection)
        else:
            return "(" + self.label + " " + " ".join((str(c[0]) for c in self.children)) + ")"

def build_tree(spines):
    adress_to_node = {}
    spines_root = []
    open_nodes = []
    
    for spine in spines:
        previous_node = None
        level = -1
        
        for label in spine["template"].split("+"):
            node = Node(label if level >= 0 else spine["pos"], spine["id"])
            open_nodes.append(node)
            if previous_node is not None:
                node.add_child(previous_node, "s")
            adress_to_node[(spine["id"], level)] = node
            
            previous_node = node
            level += 1
            
        spines_root.append(previous_node)
                
    root = Node("ROOT", 0)
    for i, spine in enumerate(spines):
        node = spines_root[i]
        if spine["head"] == 0:
            root.add_child(node, "s")
        else:
            adress_to_node[(spine["head"], spine["att_position"])].add_child(spines_root[i], spine["att_type"])
            
    while len(open_nodes) > 0:
        node = open_nodes.pop()
        
        left_children = node.left_children()
        right_children = node.right_children()
        
        if any(t == "r" for _, t in itertools.chain(left_children, right_children)):
            new_node = Node(node.label, node.projection)
            new_node.children.append(node.projection_child())
            
            if any(t == "r" for _, t in left_children):
                # won't work with a sentence which has > 10000 words
                left_most = min(left_children, key=lambda c: c[0].projection if c[1] == "r" else 10000)
                
                new_left_children = [(left_most[0], "s")]
                
                for c in left_children:
                    if c[0].projection < left_most[0].projection:
                        new_left_children.append(c)
                    elif c[0].projection > left_most[0].projection:
                        new_node.children.append(c)
                        
                left_children = new_left_children
                
            else:
                new_node.children.extend(left_children)
                left_children = []

            if any(t == "r" for _, t in right_children):
                right_most = max(right_children, key=lambda c: c[0].projection if c[1] == "r" else -1)
                
                new_right_children = [(right_most[0], "s")]
                
                for c in right_children:
                    if c[0].projection > right_most[0].projection:
                        new_right_children.append(c)
                    elif c[0].projection < right_most[0].projection:
                        new_node.children.append(c)
                        
                right_children = new_right_children
                
            else:
                new_node.children.extend(right_children)
                right_children = []
                
            node.children = [(new_node, "s")] + right_children + left_children
            open_nodes.append(new_node)
    
    
    # build discodop tree
    return root.to_discodop_tree()

spines_path = sys.argv[1]

spine_sentences = []
with open(spines_path) as f:
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

for i, spines in enumerate(spine_sentences):
    tree = build_tree(spines)
    sentence = [
        spine["word"]
        for spine
        in sorted(spines, key=lambda c: c["id"])
    ]

    print(discodop.treebank.EXPORTHEADER)
    print(discodop.treebank.writetree(tree, sentence, str(i+1), "export"))
