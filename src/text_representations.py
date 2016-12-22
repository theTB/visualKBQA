from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def embedding_lookup(text, mask, V, K, initialize=None):
    '''
    Given the text, create a embedding lookup
    Args:
        text: (tf.placeholder) the input tokens sequence
        mask: (tf.placeholder) the input sequence mask
        V: size of vocabulary
        K: size of embeddings
        initialize: (string) path to embeddings for initialization
    Returns:
        embed: (Tensor) B x maxlen x K tensor of embeddings
    '''
    pad_embedding = tf.constant(
        np.zeros((1, K), dtype=np.float32), name="pad_embedding"
    )
    embeddings = tf.concat(0,
        [pad_embedding,
         tf.Variable(tf.random_uniform([V, K], -.001, .001, dtype=tf.float32),
         name='embeddings')
        ]
    )
    embed = tf.nn.embedding_lookup(embeddings, text)

    return embed


def bidir_lstm_model(
        text, mask, V, K, nhidden, nlayers=1,
        peepholes=False, initialize=None, dropouts=None
    ):
    '''
        Args:
            text: (batchsize X maxlen) input text
            mask: (batchsize X maxlen) input mask
            V: vocabulary size
            K: word-embedding dimension
            nhiddne: number of hidden units of LSTM
            nlayers: number of layers for the LSTM
            peepholes: whether to use peephole connections
            initialize: initialization file for word embeddings
            dropouts: list of placeholders for the dropouts (size = nlayers+1)
        Returns:
            LSTM embeddings (batchsize X maxlen x hiddens)
    '''
    embeddings = embedding_lookup(text, mask, V, K, initialize)

    if dropouts:
        inputs = tf.nn.dropout(embeddings, dropouts[0])
    else:
        inputs = embeddings

    for layer in xrange(nlayers):
        fwd_cell = tf.nn.rnn_cell.LSTMCell(
            nhidden, use_peepholes=peepholes, state_is_tuple=True
        )
        bwd_cell = tf.nn.rnn_cell.LSTMCell(
            nhidden, use_peepholes=peepholes, state_is_tuple=True
        )
        lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fwd_cell, cell_bw=bwd_cell, inputs=embeddings,
            sequence_length=tf.cast(tf.reduce_sum(mask, 1), tf.int32),
            dtype=tf.float32
        )
        fwd_outputs, bwd_outputs = lstm_outputs
        inputs = tf.concat(2, [fwd_outputs, bwd_outputs])
        if dropouts:
            inputs = tf.nn.dropout(inputs, dropouts[layer+1])
        if layer < nlayers-1: nhidden = int(nhidden / 2)


    return inputs
