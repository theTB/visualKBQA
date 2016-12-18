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

import image_networks
import question_networks
import probability_networks

tf.set_random_seed(42)
np.random.seed(42)
# Why 42?: https://goo.gl/15uxva

def input_placeholders(batchsize, maxqlen, maxalen, maxrlen, maxelen, numans,
                       imageH, imageW, imageC):
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
        tf.float32, shape=(batchsize, imageH, imageW, imageC)
    )
    # for each example there are maxrlen commonsense facts
    # each fact is defined by two entites and a relation between them
    entity_relation_placeholder = tf.placeholder(
        tf.int32, shape=(batchsize, maxrlen, 3)
    )

    label_placeholder = tf.placeholder(
        tf.int32, shape=(batchsize, 1)
    )

    return ques_placeholder, ques_mask_placeholder,
           ans_placeholder, ans_mask_placeholder,
           image_placeholder,
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



if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Matrix tree parser')

    p.add_argument('--data-file', required=True, type=str,
                   help='path to parsing dataset')
    p.add_argument('--log-dir', required=True, type=str,
                   help='path to log directory')
    p.add_argument('--val-ratio', default=0.05, type=float,
                   help='ratio for validation split')
    p.add_argument('--maxqlen', default=-1, type=int, help='max question length')
    p.add_argument('--maxalen', default=-1, type=int, help='max answer length')
    p.add_argument('--batchsize', default=128, type=int, help='batchsize')
    p.add_argument('--emb-dim', default=100, type=int,
                   help='size of word embeddings')
    p.add_argument('--emb-file', default=None, type=str,
                   help='initialization file for word embeddings')
    p.add_argument('--optimizer', default='adam', type=str,
                   help=
                   'type of optimizer: sgd, adam, adadelta, adagrad, momentum')
    p.add_argument('--lr', default=0.01, type=float, help='learing rate')
    p.add_argument('--momentum', default=0.9,
                   type=float, help='momentum, only for sgd')
    p.add_argument('--decay-steps', default=1000, type=int,
                   help='steps for lr decay')
    p.add_argument('--decay-rate', default=0.96, type=float,
                   help='rate for lr decay')
    p.add_argument('--epochs', default=3, type=int,
                   help='number of train epochs')
    p.add_argument('--val-freq', default=20, type=int,
                   help='validation frequency')
    p.add_argument('--grad-clip', default=10.0, type=float,
                   help='clip l2 norm of gradients at this value')
    p.add_argument('--nhidden', default=100, type=int,
                   help='number of final hidden units, only for some models')
    p.add_argument('--nlayers', default=1, type=int,
                   help='number of layers if using LSTM model')
    p.add_argument('--dropout1', default=1, type=float,
                   help='input dropout')
    p.add_argument('--dropout2', default=1, type=float,
                   help='dropout after layer 1')

    config = p.parse_args()

    main(config)
