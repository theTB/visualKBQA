from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import cPickle
import pickle
import os
import string

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def pad_mask(X, maxlen):
    '''
    Given a list of lists X, pad each list within X to maxlen
    returns np.arrays of padded X and corresponding mask
    '''
    N = len(X)
    X_out = None
    X_out = np.zeros((N, maxlen), dtype=np.int64)
    M = np.ones((N, maxlen), dtype=np.float64)
    for i, x in enumerate(X):
        n = len(x)
        if n < maxlen:
            X_out[i, :n] = x
            M[i, n:] = 0
        else:
            X_out[i, :] = x[:maxlen]

    return X_out, M


# -------------------------------- dataset IO ------------------------------------------
def read_visual7w_dataset(qa_file, ground_annot_file):
  '''
  Read question and multiple answers for each qa pair.
  Also read grounding annotations for each qa pair.

  Input: qa_file and ground_annot_file are two files provided by the Visual7W dataset
  Return:
    vqa_data is a dictionary with keys of ['train', 'val', 'test']. Each value entry is a list,
    where each entry is a dictionary that contains:
      multiple_choices    a list consisting of 4 candidate answers
      question            a string
      filename            corresponding image name in the dataset

      *********************************** obsoleted ***************************************
      visual_concepts     a list where each entry is a dictionary containing
                            name      string, name of the visual concept
                            height    height of the bounding box around the concept in the image
                            width
                            x         left
                            y         top
                            qa_id     the question id corresponding to the concept
                          Note that, there might be no provided visual concepts in an image.
      ********************************* end obsoleted **************************************

      visual_concept      a dictionary contains
                            name      string, name of the visual concept
                            height    height of the bounding box around the concept in the image
                            width
                            x         left
                            y         top
                            qa_id     the question id corresponding to the concept
                          Note that, there might be no provided visual concepts in an image.

      split               string, train|val|test
      qa_id               index of the question
      answer              string, correct answer (the first one in multiple_choices)
      type                string, question type, one of the 7ws
  '''
  with open(qa_file, 'r') as f:
    qa_meta_data = json.load(f)['images']

  qa_data = {};
  for split in ['train', 'val', 'test']:
    qa_data[split] = []

  for qd in qa_meta_data:
    file_name = qd['filename']
    split = qd['split']
    qa_pairs = qd['qa_pairs']
    for qp in qa_pairs:
      qp['split'] = split
      qp['filename'] = file_name
      qp['multiple_choices'] = [qp['answer']] + qp['multiple_choices']
      qa_data[split].append(qp)

  # with open(ground_annot_file, 'r') as f:
  #   ground_data = json.load(f)['boxes']

  # # build a map from qa_id to image id, to group visual concepts in an image
  # qa_id_image_id = {}
  # for split in ['train', 'val', 'test']:
  #   for qa in qa_data[split]:
  #     qa_id_image_id[qa['qa_id']] = qa['image_id']

  # image_concepts = {}
  # for gd in ground_data:
  #   image_id = qa_id_image_id[gd['qa_id']]
  #   if image_id in image_concepts:
  #     image_concepts[image_id].append(gd)
  #   else:
  #     image_concepts[image_id] = [gd];

  '''
  We currently don't need this part. Concept in the question would be automatically extracted.
  '''
  # image_concepts = {}
  # for gd in ground_data:
  #   qa_id = gd['qa_id']
  #   image_concepts[qa_id] = gd

  # for split in ['train', 'val', 'test']:
  #   for idx, qa in enumerate(qa_data[split]):
  #     # image_id = qa['image_id']
  #     qa_id = qa['qa_id']
  #     if qa_id in image_concepts:
  #       visual_concepts = image_concepts[qa_id]
  #     else:
  #       visual_concetps = []
  #     qa['visual_concept'] = visual_concepts
  #     qa_data[split][idx] = qa

  return qa_data

def append_image_vqa_results(vqa_data, image_vqa_results_file, noise=0.1):
  '''
  Augment the visual question answering dataset with predicted answers from a pure image vqa model.

  Input
    vqa_data                  data returned by the read_visual7w_dataset function
    image_vqa_results_file    file path containing the image vqa predictions

  Output
    Similar vqa_data, but each of the dictionary contains more data as follows
      im_logp      a list of float values, where each entry is the average negative log likelihood
                   of all words in each candidate answer.
  '''
  with open(image_vqa_results_file, 'r') as f:
    predictions = json.load(f)

  for split in ['train', 'val', 'test']:
    # build qa_id to array index map
    qa_id_arr_idx = {}
    for idx, qa in enumerate(vqa_data[split]):
      qa_id_arr_idx[qa['qa_id']] = idx

    for p in predictions[split]:
      idx = qa_id_arr_idx[p['qa_id']]
      if split == 'train' and np.random.random() < noise:
          np.random.shuffle(p['logp'])
      vqa_data[split][idx]['im_logp'] = p['logp']
      # for check purpose
#       vqa_data[split][idx]['cquestion'] = p['question']
#       vqa_data[split][idx]['canswers'] = p['answers']

  return vqa_data

def append_kb_vqa_results(vqa_data, kb_vqa_results_file):
  '''
  Augment the visual question answering dataset with predicted answers from a pure image vqa model.

  Input
    vqa_data                  data returned by the read_visual7w_dataset function
    image_vqa_results_file    file path containing the image vqa predictions

  Output
    Similar vqa_data, but each of the dictionary contains more data as follows
      im_logp      a list of float values, where each entry is the average negative log likelihood
                   of all words in each candidate answer.
  '''
  with open(kb_vqa_results_file, 'r') as f:
    predictions = pickle.load(f)

  for split in ['train', 'val', 'test']:
    # build qa_id to array index map
    qa_id_arr_idx = {}
    for idx, qa in enumerate(vqa_data[split]):
      qa_id_arr_idx[qa['qa_id']] = idx

    for pred in predictions[split]:
      qa_id = pred['qa_id']
      idx = qa_id_arr_idx[qa_id]
      vqa_data[split][idx]['kb_logp'] = pred['kb_logp_max']
      assert(vqa_data[split][idx]['qa_id'] == qa_id)
      # for check purpose
#       vqa_data[split][idx]['cquestion'] = p['question']
#       vqa_data[split][idx]['canswers'] = p['answers']

  return vqa_data

def append_image_embeddings(vqa_data, image_embeddings_file):
  '''
  Augment the visual question answering dataset with image embeddings extracted from a pre-trained
  VGG16 CNN on ImageNet dataset.

  Input
    vqa_data                  data returned by the read_visual7w_dataset function
    image_embeddings_file    file path containing the image embeddings

  Output
    Similar vqa_data, but each of the dictionary contains more data as follows
      im_embed      a numpy array of float (1x4096)
  '''
  with open(image_embeddings_file, 'r') as f:
    image_id_embed = cPickle.load(f)

  for split in ['train', 'val', 'test']:
    for idx in xrange(len(vqa_data[split])):
      image_id = vqa_data[split][idx]['image_id']
      vqa_data[split][idx]['im_embed'] = image_id_embed[image_id]

  return vqa_data

# -------------------------------- data preprocessing ------------------------------------------
def tokenize_sentence(sentence):
  """
  Tokenize a sentence.
  Args:
    sentence: a sentence
  Returns:
    tokens: tokenized sentence, a list
  """
  tokens = str(sentence).lower().translate(None, string.punctuation).strip().split()
  return tokens

def build_vocabulary(vqa_data, word_count_threshold, verbose=False):
  """
  Build a vocabulary.
  Args:
    data: comes from _read_data_file
    word_count_threshold: words which occur less than word_count_threshold times would be converted to specil UNK tokens
  """
  count_thr = word_count_threshold

  # count up the number of words
  counts = {}
  for split in ['train', 'val']:        # test set is not allowed to build the vocabulary
    for qd in vqa_data[split]:
        for w in tokenize_sentence(qd['question']):
          counts[w] = counts.get(w, 0) + 1

        for mc in qd['multiple_choices']:
          for w in tokenize_sentence(mc):
            counts[w] = counts.get(w, 0) + 1

  # vocabulary, keep words that occur often
  vocab_list = [w for w,n in counts.iteritems() if n > count_thr]
  vocab_list = _START_VOCAB + vocab_list
  vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])

  # print some stats
  if True:  # verbose:
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))
    print('most infrequent words and their counts:')
    print('\n'.join(map(str,cw[-20:])))

    total_words = sum(counts.itervalues())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    word_freq = [n for w, n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of question lengths
    q_lengths = {}
    for split in ['train', 'val']:        # test set is not allowed to build the vocabulary
      for qd in vqa_data[split]:
        nw = len(tokenize_sentence(qd['question']))
        q_lengths[nw] = q_lengths.get(nw, 0) + 1
    max_q_len = max(q_lengths.keys())
    print('max length question in the train+val data: ', max_q_len)
    print('question length distribution (count, number of words):')
    sum_q_len = sum(q_lengths.values())
    for i in xrange(max_q_len + 1):
      print('%2d: %10d   %f%%' % (i, q_lengths.get(i,0), q_lengths.get(i,0)*100.0/sum_q_len))

    # let's look at the distribution of answers lengths as well
    a_lengths = {}
    for split in ['train', 'val']:
      for qd in vqa_data[split]:
        for mc in qd['multiple_choices']:
          nw = len(tokenize_sentence(mc))
          a_lengths[nw] = a_lengths.get(nw, 0) + 1
    max_a_len = max(a_lengths.keys())
    print('max length answer in the train+val data: ', max_a_len)
    print('answer length distribution (count, number of words):')
    sum_a_len = sum(a_lengths.values())
    for i in xrange(max_a_len + 1):
      print('%2d: %10d   %f%%' % (i, a_lengths.get(i,0), a_lengths.get(i,0)*100.0/sum_a_len))

  return vocab, max_q_len, max_a_len

def encode_question_answer(vqa_data, vocabulary):
  """Tokenize data and turn into token-ids using given vocabulary.
  Args:
    data: a list of dict object.
    vocabulary: pre-built vocabulary, a dict
  Returns:
    data: a list of dict objects, where token ids are stored in data[i]['token_ids']
  """
  for split in ['train', 'val', 'test']:
    for i, qd in enumerate(vqa_data[split]):
      words = tokenize_sentence(qd['question'])
      vqa_data[split][i]['question_tk_ids'] = [vocabulary.get(w, UNK_ID) for w in words]
      vqa_data[split][i]['multiple_choices_tk_ids'] = [[] for _ in qd['multiple_choices']]

      for j, mc in enumerate(qd['multiple_choices']):
        words = tokenize_sentence(mc)
        vqa_data[split][i]['multiple_choices_tk_ids'][j] = [vocabulary.get(w, UNK_ID) for w in words]

  return vqa_data

def preprocess_raw_vqa_data(vqa_data, word_count_threshold, verbose=False):
  vocab, max_q_len, max_a_len = build_vocabulary(vqa_data, word_count_threshold, verbose)
  vqa_data = encode_question_answer(vqa_data, vocab)

  return vqa_data, vocab, max_q_len, max_a_len

def vqa_data_iterator(vqa_data, split, batch_size, max_q_len, max_a_len, do_permutation=False):
  """
  Iterate on the raw vqa data.
  Args:
    vqa_data:         output of preprocess_raw_vqa_data
    split:            'train'|'val'|'test'
    batch_size:       int, the batch size
    max_q_len         maximum length of a question
    max_a_len         maximum length of an answer
    do_permutation    if to randomly shuffle the four candidate answers
  Returns:
    A dictionary has following fields:
      imbed = data['image_embed']
      ques = data['question']
      qmask = data['question_mask']
      ans = data['answers']
      ans_mask = data['answers_mask']
      im_logp = data['im_vqa_logp']
      kb_logp = data['kb_vqa_logp']
      label = data['labels']
  """
  data_len = len(vqa_data[split])
  batch_len = data_len // batch_size

  np.random.shuffle(vqa_data[split])

  im_embed_dim = 4096
  num_ans = 4

  for i in range(batch_len):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size

    im_embed = np.zeros((batch_size, im_embed_dim), dtype=np.float32)
    im_logp = np.zeros((batch_size, num_ans), dtype=np.float32)
    kb_logp = np.zeros((batch_size, num_ans), dtype=np.float32)
    label = np.zeros((batch_size, num_ans), dtype=np.float32)

    # max_q_len = -1
    # max_a_len = -1

    # for idx, data in enumerate(vqa_data[start_idx:end_idx]):
    #   max_q_len = max(max_q_len, len(data['question_tk_ids']))
    #   for ans in data['multiple_choices_tk_ids']:
    #     max_a_len = max(max_a_len, len(ans))

    ques = np.zeros((batch_size, max_q_len), dtype=np.int32)
    ques_mask = np.zeros((batch_size, max_q_len), dtype=np.int32)
    ans = np.zeros((batch_size, num_ans, max_a_len), dtype=np.int32)
    ans_mask = np.zeros((batch_size, num_ans, max_a_len), dtype=np.int32)

    for idx, data in enumerate(vqa_data[split][start_idx:end_idx]):
      im_embed[idx,:] = data['im_embed']
      im_logp[idx,:] = data['im_logp']
      kb_logp[idx,:] = data['kb_logp']
      label[idx,:] = [1, 0, 0, 0]

      q_len = len(data['question_tk_ids'])
      ques[idx, :q_len] = data['question_tk_ids']
      ques_mask[idx, :q_len] = 1      # FIX ME!!! potentially a bug
      for jdx, choice in enumerate(data['multiple_choices_tk_ids']):
        a_len = len(choice)
        ans[idx, jdx, :a_len] = choice
        ans_mask[idx, jdx, :a_len] = 1

      if do_permutation:
        rand_idx = [0, 1, 2, 3]
        np.random.shuffle(rand_idx)
        im_logp = im_logp[:, rand_idx]
        kb_logp = kb_logp[:, rand_idx]
        label = label[:, rand_idx]
        ans = ans[:, rand_idx, :]
        ans_mask = ans_mask[:, rand_idx, :]

    yield {'image_embed': im_embed,
           'question': ques,
           'question_mask': ques_mask,
           'answers': ans,
           'answers_mask': ans_mask,
           'im_logp': im_logp,
           'kb_logp': kb_logp,
           'label': label}
