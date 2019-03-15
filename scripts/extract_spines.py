#!/bin/python3

import sys

import discodop.treebank
import discodop.heads

treebank_path = sys.argv[1]
headrules_path = sys.argv[2]

reader = discodop.treebank.NegraCorpusReader(treebank_path)
trees = reader.trees()
sentences = reader.sents()

headrules = discodop.heads.readheadrules(headrules_path)

def extract_spines(tree, radj=True):
    node_level = {}
    node_allow_r_adj = {}
    node_projection = {}
    
    for subtree in tree.subtrees(lambda n: n and not isinstance(n[0], discodop.treebank.Tree)):
        subtree_save = subtree
        
        # check if this is correct for computing the word index
        #projection = tree.leaves().index(subtree[0]) + 1
        projection = subtree[0] + 1
        level = 0
        
        #node_level[subtree.treeposition] = level
        node_projection[subtree.treeposition] = projection
        
        previous_label = subtree.label
        while discodop.heads.ishead(subtree):
            subtree = subtree.parent
            
            if previous_label != subtree.label or not radj:
                level += 1
                previous_label = subtree.label
                node_allow_r_adj[subtree.treeposition] = False
            else:
                node_allow_r_adj[subtree.treeposition] = True
            
            node_level[subtree.treeposition] = level - 1 # minus because Carreras did not count the POS
            node_projection[subtree.treeposition] = projection
            
            previous_label = subtree.label
                    
    spines = []
    for subtree in tree.subtrees(lambda n: n and not isinstance(n[0], discodop.treebank.Tree)):
        p = subtree.treeposition
        
        spine = {
            "id": node_projection[p],
            "pos": subtree.label,
            "template": "*",
            "head": 0,
            "att_position": 0,
            "att_type": "s"
        }

        previous_label = subtree.label
        while discodop.heads.ishead(subtree):
            subtree = subtree.parent
            if subtree.treeposition == ():
                break
            
            if previous_label != subtree.label or not radj:
                spine["template"] += "+" + subtree.label
                
            previous_label = subtree.label
                
        if subtree.parent is not None:
            parent = subtree.parent
            parent_p = subtree.parent.treeposition
            
            spine["head"] = node_projection[parent_p]
            spine["att_position"] = node_level[parent_p]
            
            if radj and node_allow_r_adj[parent_p] and \
                (
                    (subtree.right_sibling is not None and discodop.heads.ishead(subtree.right_sibling))
                    or
                    (subtree.left_sibling is not None and discodop.heads.ishead(subtree.left_sibling))
                ):
                spine["att_type"] = "r"

            
        
        spines.append(spine)
        
    return spines

for k in trees:
    discodop.heads.applyheadrules(trees[k], headrules)

for k in trees:
    tree = trees[k]
    sentence = sentences[k]
    spines = extract_spines(tree)

    spines.sort(key=lambda s: s["id"])
    for spine in spines:
        print("%i\t%s\t%s\t%s\t%i\t%s\t%s"%(
	    spine["id"],
            sentence[spine["id"] - 1],
	    spine["pos"],
	    spine["template"],
	    spine["head"],
	    spine["att_position"] if spine["head"] != 0 else 0,
	    spine["att_type"] if spine["head"] != 0 else "s"
	))

    print()
