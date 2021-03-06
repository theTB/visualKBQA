from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def simple_mlp(nhidden, nlayers, noutputs=1, *vectors):
    inputs = tf.concat(1, list(vectors))
    shape = inputs.get_shape().as_list()
    print("Prob Net input: ", shape)
    hidden = nhidden
    if nlayers == 1:
        print("Probability has only one layer")
        hidden = noutputs
    layer_inputs = inputs
    for layer in xrange(nlayers):
        shape = layer_inputs.get_shape().as_list()
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
        if layer < nlayers - 1:
            layer_inputs = tf.nn.relu(affine)
        else:
            # output = tf.nn.sigmoid(affine)
            output = tf.nn.relu(affine)
        if layer < nlayers - 2:
            hidden = int(hidden / 2)
        else:
            hidden = noutputs

    return output
