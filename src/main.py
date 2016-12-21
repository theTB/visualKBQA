from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import sys
from itertools import izip
import time
import cPickle

import image_networks
import question_networks
import probability_networks
import read_data

tf.set_random_seed(42)
np.random.seed(42)
# Why 42?: https://goo.gl/15uxva

# revised by Huaizu
# create a place holder for image embedding, not the entire image (assume the image CNN is fixed)
def input_placeholders(batchsize, maxqlen, maxalen, maxrlen, maxelen, numans,
                       image_embed_dim):
    '''
    Placeholders for
        input questions, question mask
        input answers, answer masks
        correct answers
        images
        entity pairs and relation between them (3 ints: e1, e2, r)
    '''
    ques_placeholder = tf.placeholder(tf.int32, shape=(batchsize, maxqlen))
    ques_mask_placeholder = tf.placeholder(
        tf.int32, shape=(batchsize, maxqlen)
    )
    ans_placeholder = tf.placeholder(
        tf.int32, shape=(batchsize, numans, maxalen)
    )
    ans_mask_placeholder = tf.placeholder(
        tf.int32, shape=(batchsize, numans, maxalen)
    )

    image_placeholder = tf.placeholder(
        tf.float32, shape=(batchsize, image_embed_dim)
    )

    # for each example there are maxrlen commonsense facts
    # each fact is defined by two entites and a relation between them
    # entity_relation_placeholder = tf.placeholder(
    #     tf.int32, shape=(batchsize, maxrlen, 3)
    # )

    im_vqa_logp = tf.placeholder(tf.float32, shape=[batchsize, numans])
    kb_vqa_lopg = tf.placeholder(tf.float32, shape=[batchsize, numans])

    label_placeholder = tf.placeholder(
        tf.float32, shape=(batchsize, numans)
    )

    return ques_placeholder, ques_mask_placeholder,
           ans_placeholder, ans_mask_placeholder,
           image_embed_placeholder,
           label_placeholder


def get_optimizer(config, step):
    if config.optimizer == 'sgd':
        learning_rate = tf.train.exponential_decay(
            config.lr, step, config.decay_steps,
            config.decay_rate, staricase=True
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif config.optimizer == 'adam':
        # not sure if dcaying learning_rate is a good idea here
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    elif config.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(config.lr, step)
    elif config.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=config.lr)
    elif config.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(config.lr, config.momentum)
    else:
        print ("Incorrect Optimizer specified")
        sys.exit()

    return optimizer


def main(config):
    # get data
    cache_file = os.path.join(config.cache_dir, 'vqa_data.pkl')
    if not os.path.exists(cache_file):
      vqa_data = read_data.read_visual7w_dataset(config.data_file, '')
      vqa_data = read_data.append_image_vqa_results(vqa_data, config.im_vqa_file)
      vqa_data = read_data.append_image_embeddings(vqa_data, config.im_embed_file)
      vqa_data, vocab, maxqlen, maxalen = preprocess_raw_vqa_data(vqa_data, config.word_cnt_thresh, verbose=config.verbose)

      # assume the KB scores are available
      for split in ['train', 'val', 'test']:
        for i in xrange(len(vqa_data[split])):
          vqa_data[split][i]['kb_logp'] = -np.log(np.random.random_sample(4) * 0.1 + 0.85)   #[0.1, 0.95)

      with open(cache_file, 'w') as f:
        cPickle.dump((vqa_data, vocab, maxqlen, maxalen), f)
    else:
      with open(cache_file, 'r') as f:
        vqa_data, vocab, maxqlen, maxalen = cPickle.load(f)

    # define the graph of the probability network
    # e_I       embedding of image
    # e_Q       embedding of question
    # e_As      list of embeddings of each answer
    # FIX ME!!! ugly!!!
    ques_placeholder, ques_mask_placeholder, ans_placeholder, ans_mask_placeholder, pre_image_embed_placeholder, label_placeholder = input_placeholders(
      config.batchsize, maxqlen, maxalen, maxrlen=1, maxelen=1, numans=4, image_embed_dim=4096)

    # image embedding
    image_embed_w = tf.Variable(tf.zeros([4096, emb_dim]))
    image_embed_b = tf.Variable(tf.zeros([emb_dim]))
    image_embed = tf.nn.relu(tf.matmul(pre_image_embed_placeholder, image_embed_w) + image_embed_b)

    # question embedding
    V = len(vocab.keys())
    K = emb_dim
    nhidden = emb_dim
    ques_embed = qa_text_networks.question_lstm(ques_placeholder, ques_mask_placeholder, V, K, nhidden)

    # answers embedding
    ans_embed = qa_text_networks.answer_lstm(ans_placeholder, ans_mask_placeholder, V, K, nhidden)

    # probability network
    p = probability_networks.simple_mlp(emb_dim, 1, [image_embed, ques_embed, ans_embed])

    # combine image- and kb-vqa results
    
    logits = tf.mul(p, im_vqa_logp) + tf.mul(1 - p, kb_vqa_lopg)

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, label_placeholder))

    train_step = get_optimizer(config, step)

    while data in read_data.vqa_data_iterator(vqa_data, 'train', config.batchsize):
      imbed = data['image_embed']
      ques = data['question']
      ans = data['answers']
      im_vqa_logp = data['im_vqa_logp']
      kb_vqa_lopg = data['kb_vqa_lopg']
      # train_step.run(feed_dict={})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Matrix tree parser')

    p.add_argument('--data_file', required=True, type=str,
                   help='path to parsing dataset')
    p.add_argument('--log_dir', required=True, type=str,
                   help='path to log directory')
    # -------- Huaizu: we currently don't need this parameter as train/val/test split has been done -------
    # p.add_argument('--val-ratio', default=0.05, type=float,
    #                help='ratio for validation split')
    p.add_argument('--maxqlen', default=-1, type=int, help='max question length')
    p.add_argument('--maxalen', default=-1, type=int, help='max answer length')
    p.add_argument('--batchsize', default=128, type=int, help='batchsize')
    p.add_argument('--emb_dim', default=100, type=int,
                   help='size of word embeddings')
    p.add_argument('--emb_file', default=None, type=str,
                   help='initialization file for word embeddings')
    p.add_argument('--optimizer', default='adam', type=str,
                   help=
                   'type of optimizer: sgd, adam, adadelta, adagrad, momentum')
    p.add_argument('--lr', default=0.01, type=float, help='learing rate')
    p.add_argument('--momentum', default=0.9,
                   type=float, help='momentum, only for sgd')
    p.add_argument('--decay_steps', default=1000, type=int,
                   help='steps for lr decay')
    p.add_argument('--decay_rate', default=0.96, type=float,
                   help='rate for lr decay')
    p.add_argument('--epochs', default=3, type=int,
                   help='number of train epochs')
    p.add_argument('--val_freq', default=20, type=int,
                   help='validation frequency')
    p.add_argument('--grad_clip', default=10.0, type=float,
                   help='clip l2 norm of gradients at this value')
    p.add_argument('--nhidden', default=100, type=int,
                   help='number of final hidden units, only for some models')
    p.add_argument('--nlayers', default=1, type=int,
                   help='number of layers if using LSTM model')
    p.add_argument('--dropout1', default=1, type=float,
                   help='input dropout')
    p.add_argument('--dropout2', default=1, type=float,
                   help='dropout after layer 1')

    # added by Huaizu
    p.add_argument('--cache_dir', required=True, type=str,
                   help='path to cache directory')
    p.add_argument('--word_cnt_thresh', required=True, type=int,
                   help='a threshold of word occurences, below the threshold, a word will be replaced by UNK token')
    p.add_argument('--verbose', required=False, default=False, type=bool,
                   help='if to display intermediate results')

    config = p.parse_args()

    main(config)
