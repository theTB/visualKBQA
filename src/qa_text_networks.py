from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import text_representations

def question_lstm(text, mask, V, K, nhidden, nlayers=1,
                  peepholes=False, initialize=None, dropouts=None):
    lstm_embeddings = text_representations.bidir_lstm_model(
        text, mask, V, K, nhidden, nlayers,
        peepholes, initialize, dropouts
    )
    # output is (B x M x K)
    sum_states = tf.reduce_sum(lstm_embeddings, 1)
    lens = tf.reduce_sum(mask, 1, keep_dims=True)
    # average of the hidden state outputs
    question_vectors = sum_states / lens

    return question_vectors


def answer_lstm(text, mask, V, K, nhidden, nlayers=1,
                  peepholes=False, initialize=None, dropouts=None):
    '''
    text and mask here have dims (batchsize X numans X maxalen)
    '''
    shape = tf.shape(text)
    # reshape to (batchsize*numans X maxalen)
    batch_text = tf.reshape(text, (shape[0]*shape[1], shape[2]))
    batch_mask = tf.reshape(text, (shape[0]*shape[1], shape[2]))
    batch_lstm_embeddings = text_representations.bidir_lstm_model(
        batch_text, batch_mask, V, K, nhidden, nlayers,
        peepholes, initialize, dropouts
    )
    # output is (B*Na x M x hidden)
    sum_states = tf.reduce_sum(lstm_embeddings, 1)
    lens = tf.reduce_sum(batch_mask, 1, keep_dims=True)
    # average of the hidden state outputs
    batch_answer_vectors = sum_states / lens
    # output is (B*Na x K)
    output_shape = tf.shape(batch_answer_vectors)
    answer_vectors = tf.reshape(
        batch_answer_vectors, (shape[0], shape[1], output_shape[-1])
    )

    return answer_vectors
