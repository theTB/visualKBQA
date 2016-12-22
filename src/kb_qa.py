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
stopwords = nltk.corpus.stopwords.words('english')

def chunker(text):
    tokens = nltk.word_tokenize(text.strip())
    pos_tags = nltk.pos_tag(tokens)
    parse_tree = chunk_parser.parse(pos_tags)
    chunked_tokens = []
    # for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
    # print the noun phrase as a list of part-of-speech tagged words
    print(parse_tree)
    for node in parse_tree:
        if type(node) == nltk.tree.Tree:
            chunk = node.leaves()
            word = "_".join(map(lambda x: x[0], chunk))
            # print(word)
        else:
            word = node[0]
        chunked_tokens.append(word)

    return chunked_tokens


def extract_entities(text):
    chunked_tokens = chunker(text)
    tokens = map(lambda x: x.strip().lower(), chunked_tokens)
    tokens = filter(lambda x: x not in stopwords, tokens)
    return tokens


def score_fn(Rel, We, Weight, Offset, words, rel, evalType='max'):
    fn = lambda x1, x2: score(
        x1, x2, words, We, rel, Rel, Weight, Offset, evalType
    )
    score_fn = lambda x1, x2: max(fn(x1, x2), fn(x2, x1))
    return score_fn


def kb_scores(vqa_data, score_fn, outname):
    all_scores = []
    for qa_example in vqa_data:
        qid = qa_example['qa_id']
        question = qa_example['question']
        answers = qa_example['multiple_choices']
        q_ents = extract_entities(question)
        ans_ents = map(lambda x: extract_entities(x), answers)
        scores = []
        for a_ents in ans_ents:
            mscore = 0
            for ent in a_ents:
                # score answer entities with question entities and take max
                s = max(lambda x: score_fn(x, ent), q_ents)
                mscore = max(mscore, s)
            scores.append(mscore)
        ans = {'qa_id': qid, 'multiple_choices': all_scores}
        all_scores.append(ans)

    with open(outname, 'w+') as f:
        pickle.dump(all_scores, f)

    return all_scores








if __name__ == "__main__":
    p = argparse.ArgumentParser(description='ConceptNet scores for QA')

    p.add_argument('--model', required=True, type=str,
                   help='path to conceptnet pickled model')
    p.add_argument('--vqa', required=True, type=str,
                   help='path to pickled vqa data')
    p.add_argument('--output', required=True, type=str,
                   help='path to output file')

    config = p.parse_args()

    with open(config.model, 'r') as f:
        model = pickle.load(f)

    Rel = model['rel']
    We = model['embeddings']
    Weight = model['weight']
    Offset = model['bias']
    words = model['words_name']
    rel = model['rel_name']
