rom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import read_data

import argparse
import pickle
from demo_bilinear import score

import nltk

grammar = '''
            NP: {<DT|PP\$>?<JJ>*<NN>}
                {<NNP>+}
                {<NN>+}
          '''
chunk_parser = nltk.RegexpParser(grammar)

def traverse(t):
    # print(t)
    try:
        t.label()
    except AttributeError:
        return
    if t.label() == 'NP':
        print(t)
    else:
        for child in t:
            traverse(child)

def extract_entities(text):
    tokens = nltk.word_tokenize(text.strip())
    pos_tags = nltk.pos_tag(tokens)
    parse_tree = chunk_parser.parse(pos_tags)
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
    # print the noun phrase as a list of part-of-speech tagged words
        print subtree.leaves()






def score_pairs():



if __name__ == "__main__":
    p = argparse.ArgumentParser(description='ConceptNet scores for QA')

    p.add_argument('--model', required=True, type=str,
                   help='path to conceptnet scoring model')
    p.add_argument('--output', required=True, type=str,
                   help='path to output file')

    config = p.parse_args()

    Rel = model['rel']
    We = model['embeddings']
    Weight = model['weight']
    Offset = model['bias']
    words = model['words_name']
    rel = model['rel_name']
