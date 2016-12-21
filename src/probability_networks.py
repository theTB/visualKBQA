from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def simple_mlp(nhidden, nlayers, noutputs=4, *vectors):
    inputs = tf.concat(1, list(vectors))
    shape = tf.shape(inputs)
    hidden = nhidden
    if nlayers == 1:
        assert(hidden == 1)
    layer_inputs = inputs
    for layer in xrange(nlayers):
        W = tf.Variable(
            tf.random_normal(
                [shape[1], hidden], mean=0, stddev=0.01, dtype=tf.float32
            ),
            name='W'+str(layer)
        )
        b = tf.Variable(
            tf.random_normal(
                [1, hidden], mean=0, stddev=0.01, dtype=tf.float32
            ),
            name="b"+str(layer)
        )
        affine = tf.matmul(layer_inputs, W) + b
        # if layer < nlayers - 1:
        layer_inputs = tf.nn.relu(affine)
        if layer < nlayers - 2:
            hidden = int(hidden / 2)
        else:
            hidden = noutputs

    return layer_inputs
