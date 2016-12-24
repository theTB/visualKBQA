from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import read_data
import os
import re

import argparse
import pickle
import cPickle
from demo_bilinear import score

from joblib import Parallel, delayed
import multiprocessing

import nltk

# MODEL_PATH = "../Bilinear_cetrainSize300frac1.0dSize200relSize150acti0.001.1e-05.800.RAND.tanh.txt19.pickle"
MODEL_PATH = "/Users/tbansil/Downloads/ckbc-demo/Bilinear_cetrainSize300frac1.0dSize200relSize150acti0.001.1e-05.800.RAND.tanh.txt19.pickle"
with open(MODEL_PATH, 'r') as f:
    model = pickle.load(f)

print("Model loaded")

Rel = model['rel']
We = model['embeddings']
Weight = model['weight']
Offset = model['bias']
words = model['words_name']
rel = model['rel_name']
print(rel)
del_rels = ['HasPainIntensity','HasPainCharacter','LocationOfAction','LocatedNear',
'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
'NotHasA','NotIsA','NotHasProperty','NotCapableOf']

for del_rel in del_rels:
    del rel[del_rel.lower()]


grammar = '''
            NP: {<DT|PP\$>?<JJ>*<NN>}
                {<NNP>+}
                {<NN>+}
          '''
chunk_parser = nltk.RegexpParser(grammar)
stopwords = nltk.corpus.stopwords.words('english')
stopwords += ['.', '.', ',', '/', '?', "'", ":", ";", "!", "many", "'s"]

def chunker(text):
    tokens = nltk.word_tokenize(text.strip())
    pos_tags = nltk.pos_tag(tokens)
    parse_tree = chunk_parser.parse(pos_tags)
    chunked_tokens = []
    # for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
    # print the noun phrase as a list of part-of-speech tagged words
    # print(parse_tree)
    for node in parse_tree:
        if type(node) == nltk.tree.Tree:
            chunk = node.leaves()
            if len(chunk) > 1 and chunk[0][0].lower() in stopwords:
                chunk = chunk[1:]
            word = "_".join(map(lambda x: re.sub('-', '_', x[0]), chunk))
            # print(word)
        else:
            word = re.sub('-', '_', node[0])
        chunked_tokens.append(word)

    return chunked_tokens


def extract_entities(text):
    chunked_tokens = chunker(text)
    tokens = map(lambda x: x.strip().lower(), chunked_tokens)
    tokens = filter(lambda x: x not in stopwords, tokens)
    return tokens


def score_fn(Rel, We, Weight, Offset, words, rel, evalType='max'):
    fn = lambda x1, x2: max([v for k, v in score(
        x1, x2, words, We, rel, Rel, Weight, Offset, evalType
    )])
    scorer = lambda x1, x2: max(fn(x1, x2), fn(x2, x1))
    return scorer


def kb_scores_single(qa_example, verbose=False):
    qid = qa_example['qa_id']
    question = qa_example['question']
    answers = qa_example['multiple_choices']
    q_ents = extract_entities(question)
    ans_ents = map(lambda x: extract_entities(x), answers)
    scores_max = []
    scores_mean = []
    all_score_data = []
    if verbose:
        print(question)
        print(q_ents)
    for i, a_ents in enumerate(ans_ents):
        # a_ents are the entities in each answer
        if verbose:
            print("\t", answers[i])
        ans_ent_scores = []
        maxscore = 1e-8
        meanscore = 0.
        ent_cnt = 0
        for ent in a_ents:
            # score answer entities with question entities and take max
            # sc = map(
            #     lambda x: ,
            #     [e for e in q_ents if e != ent]
            # )
            sc = []
            s = 1e-8
            for e in q_ents:
                if e == ent: continue
                s1 = max([v for k, v in score(e, ent, words, We, rel, Rel, Weight, Offset, 'max')])
                s2 = max([v for k, v in score(ent, e, words, We, rel, Rel, Weight, Offset, 'max')])
                sc.append((e, ent, max(s1,s2)))
                # score for this answer entity is max across all question entities
                s = max(s, max(s1, s2))
            if len(sc) > 0:
                ent_cnt += 1
            if verbose:
                j = 0
                for e in q_ents:
                    if e != ent:
                        print("\t\tscore %s %s " %(e, ent), sc[j])
                        j += 1
            ans_ent_scores += sc
            # print(s)
            # print(mscore)
            maxscore = max(maxscore, s)
            meanscore += s
        scores_max.append(-np.log(maxscore))
        if ent_cnt > 0:
            scores_mean.append(-np.log(meanscore / ent_cnt))
        else:
            scores_mean.append(-np.log(1e-8))
        all_score_data.append(ans_ent_scores)
        if verbose:
            print(question, " ", answers[i], " score: ", maxscore, meanscore / max(1,ent_cnt))

    return qid, question, answers, all_score_data, scores_max, scores_mean

def kb_scores(vqa_data, outname, dsplit='all', verbose=True, pool='max'):
    all_scores = {}
    if dsplit == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [dsplit]
    num_cores=multiprocessing.cpu_count()
    print("Using %d CPUs" % num_cores)

    all_scores = {}
    for split in splits:
        total = len(vqa_data[split])
        print(split + " set, total ", total)
        all_scores[split] = []
        results = Parallel(n_jobs=num_cores)(
            delayed(kb_scores_single)(qa_example)
            for qa_example in vqa_data[split]
        )

        for qid, question, answers, all_score_data, scores_max, scores_mean in results:
            ans = {'qa_id': qid,
                   'multiple_choices': answers,
                   'question': question,
                   'kb_logp_max': scores_max,
                   'kb_logp_mean': scores_mean,
                   'all_data': all_score_data
                   }
            all_scores[split].append(ans)

        print(split + " done.")
        with open(outname + "."+split+".pkl", 'w+') as f:
            pickle.dump({split: all_scores[split]}, f)


    with open(outname, 'w+') as f:
        pickle.dump(all_scores, f)

    return all_scores


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='ConceptNet scores for QA')

    # p.add_argument('--model', required=True, type=str,
    #                help='path to conceptnet pickled model')
    p.add_argument('--vqa', required=True, type=str,
                   help='path to pickled vqa data')
    p.add_argument('--output', required=True, type=str,
                   help='path to output file')
    p.add_argument('--save-small', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--split', default='all', type=str)

    config = p.parse_args()

    # with open(config.model, 'r') as f:
    #     model = pickle.load(f)
    #
    # print("Model loaded")
    #
    # Rel = model['rel']
    # We = model['embeddings']
    # Weight = model['weight']
    # Offset = model['bias']
    # words = model['words_name']
    # rel = model['rel_name']
    # print(rel)
    # del_rels = ['HasPainIntensity','HasPainCharacter','LocationOfAction','LocatedNear',
    # 'DesireOf','NotMadeOf','InheritsFrom','InstanceOf','RelatedTo','NotDesires',
    # 'NotHasA','NotIsA','NotHasProperty','NotCapableOf']
    #
    # for del_rel in del_rels:
    #     del rel[del_rel.lower()]

    with open(config.vqa, 'r') as f:
        # vqa_data, vocab, maxqlen, maxalen = cPickle.load(f)
        vqa_data = cPickle.load(f)
    if config.save_small:
        if not os.path.exists(config.vqa+".small.pkl"):
            small = {}
            small['train'] = vqa_data['val']
            with open(config.vqa + ".small.pkl", 'w+') as sf:
                cPickle.dump(small, sf)
    print("Data loaded")

    # score_func = score_fn(Rel, We, Weight, Offset, words, rel, 'max')

    print("Getting scores")
    all_scores = kb_scores(vqa_data, config.output, config.split, config.verbose)
