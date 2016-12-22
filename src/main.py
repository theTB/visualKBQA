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
import qa_text_networks
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

    image_embed_placeholder = tf.placeholder(
        tf.float32, shape=(batchsize, image_embed_dim)
    )

    # for each example there are maxrlen commonsense facts
    # each fact is defined by two entites and a relation between them
    # entity_relation_placeholder = tf.placeholder(
    #     tf.int32, shape=(batchsize, maxrlen, 3)
    # )

    im_vqa_logp = tf.placeholder(tf.float32, shape=[batchsize, numans])
    kb_vqa_logp = tf.placeholder(tf.float32, shape=[batchsize, numans])

    label_placeholder = tf.placeholder(
        tf.float32, shape=(batchsize, numans)
    )

    return ques_placeholder, ques_mask_placeholder, \
           ans_placeholder, ans_mask_placeholder, \
           image_embed_placeholder, \
           im_vqa_logp, kb_vqa_logp, \
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
      print("Loading and Processing Data from raw...")
      vqa_data = read_data.read_visual7w_dataset(config.data_file, '')
      vqa_data = read_data.append_image_vqa_results(vqa_data, config.im_vqa_file)
      vqa_data = read_data.append_image_embeddings(vqa_data, config.im_embed_file)
      vqa_data, vocab, maxqlen, maxalen = read_data.preprocess_raw_vqa_data(vqa_data, config.word_cnt_thresh, verbose=config.verbose)

      # assume the KB scores are available
      for split in ['train', 'val', 'test']:
        for i in xrange(len(vqa_data[split])):
          vqa_data[split][i]['kb_logp'] = -np.log(np.random.random_sample(4) * 0.1 + 0.45)   #[0.1, 0.55)
          vqa_data[split][i]['kb_logp'][0] = -np.log(0.95)

      with open(cache_file, 'w') as f:
        cPickle.dump((vqa_data, vocab, maxqlen, maxalen), f)
    else:
      print("Loading Data from cache...")
      with open(cache_file, 'r') as f:
        vqa_data, vocab, maxqlen, maxalen = cPickle.load(f)

    print("Data loaded")
    # define the graph
    (ques_placeholder, ques_mask_placeholder, ans_placeholder,
     ans_mask_placeholder, pre_image_embed_placeholder,
     im_vqa_logp, kb_vqa_logp,
     label_placeholder) = input_placeholders(
        config.batchsize, maxqlen, maxalen,
        maxrlen=1, maxelen=1, numans=4, image_embed_dim=4096
    )

    # image embedding
    image_embed_w = tf.Variable(tf.zeros([4096, config.nhidden]))
    image_embed_b = tf.Variable(tf.zeros([config.nhidden]))
    image_embed = tf.nn.relu(
        tf.matmul(pre_image_embed_placeholder, image_embed_w) + image_embed_b
    )

    # question embedding
    V = len(vocab.keys())
    K = config.emb_dim
    nhidden = config.nhidden
    ques_embed = qa_text_networks.question_lstm(
        ques_placeholder, ques_mask_placeholder, V, K, nhidden
    )

    # answers embedding
    ans_embed = qa_text_networks.answer_lstm(
        ans_placeholder, ans_mask_placeholder, V, K, nhidden
    )
    ans_shape = tf.shape(ans_embed)

    # probability network
    print(image_embed.get_shape())
    print(ques_embed.get_shape())
    print(ans_embed.get_shape())
    p = probability_networks.simple_mlp(
        config.nhidden, config.nlayers, 4,
        image_embed, ques_embed, *tf.unpack(ans_embed, axis=1)
    )
    print(p.get_shape())

    # combine image- and kb-vqa results
    # logits = tf.mul(p, im_vqa_logp) + tf.mul(1 - p, kb_vqa_logp)
    logits = p * tf.exp(-im_vqa_logp) + (1 - p) * tf.exp(-kb_vqa_logp)

    # loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, label_placeholder)
    )

    # optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')
    incr_step = tf.assign(global_step, global_step + 1)
    optimizer = get_optimizer(config, global_step)

    def get_train_op(loss, optimizer):
        variables = tf.trainable_variables()
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        # var_names = [v.name for v in variables]
        grad_var_norms = [(tf.global_norm([gv[1]]), tf.global_norm([gv[0]]))
                          for gv in grads_and_vars]
        capped_grads, global_norm = tf.clip_by_global_norm(
            [gv[0] for gv in grads_and_vars], config.grad_clip
        )
        capped_grads_and_vars = [(capped_grads[i], gv[1])
                                 for i, gv in enumerate(grads_and_vars)]
        # norms of gradients for debugging
        grad_norms = [tf.sqrt(tf.reduce_sum(tf.square(grad)))
                      for grad, _ in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_grads_and_vars)
        return train_op, grad_norms

    train_op, grad_norms = get_train_op(cross_entropy, optimizer)

    # assume the first one is always the correct one
    def eval_accuracy(logits, labels):
        pred = np.argmax(logits, axis=1)
        # acc = np.mean(np.equal(pred, labels))
        acc = np.sum(np.equal(pred, 0))
        return acc

    session = tf.Session()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    init_op = tf.initialize_all_variables()
    session.run(init_op)

    print("Starting Training")

    for epoch in xrange(config.epochs):
        for data in read_data.vqa_data_iterator(
            vqa_data, 'train', config.batchsize, maxqlen, maxalen, do_permutation=True
        ):
            imbed = data['image_embed']
            ques = data['question']
            qmask = data['question_mask']
            ans = data['answers']
            ans_mask = data['answers_mask']
            im_logp = data['im_logp']
            kb_logp = data['kb_logp']
            label = data['label']
            feed_dict = {
                ques_placeholder: ques,
                ques_mask_placeholder: qmask,
                ans_placeholder: ans,
                ans_mask_placeholder: ans_mask,
                pre_image_embed_placeholder: imbed,
                im_vqa_logp: im_logp,
                kb_vqa_logp: kb_logp,
                label_placeholder: label
            }

            _, batch_loss, step, prs = session.run(
                [train_op, cross_entropy, incr_step, p],
                feed_dict=feed_dict
            )
            # print("probs:")
            # print(prs)
            # print("imqa:")
            # print(im_logp)
            # print("kbqa:")
            # print(kb_logp)
            # print("labels:")
            # print(label)

            print("Step %d, Loss: %.3f" % (step, batch_loss))
            if step % config.val_freq == 0:
                # save model
                saver.save(
                    session,
                    config.log_dir + "/model-epoch%d-step%d" % (epoch, step)
                )
                # Do some validation
                acc = 0
                cnt = 0
                for val_data in read_data.vqa_data_iterator(
                    vqa_data, 'val', config.batchsize, maxqlen, maxalen, do_permutation=False
                ):
                  imbed = val_data['image_embed']
                  ques = val_data['question']
                  qmask = val_data['question_mask']
                  ans = val_data['answers']
                  ans_mask = val_data['answers_mask']
                  im_logp = val_data['im_logp']
                  kb_logp = val_data['kb_logp']
                  label = val_data['label']
                  feed_dict = {
                      ques_placeholder: ques,
                      ques_mask_placeholder: qmask,
                      ans_placeholder: ans,
                      ans_mask_placeholder: ans_mask,
                      pre_image_embed_placeholder: imbed,
                      im_vqa_logp: im_logp,
                      kb_vqa_logp: kb_logp,
                      label_placeholder: label
                  }

                  logit_val = session.run(logits, feed_dict=feed_dict)
                  accuracy = eval_accuracy(logit_val, label)
                  acc += accuracy
                  cnt += label.shape[0]

                print("Acccucy of validation: %f" % (acc / cnt))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Image QA using ConceptNet')

    p.add_argument('--data_file', required=True, type=str,
                   help='path to parsing dataset')
    p.add_argument('--log_dir', required=True, type=str,
                   help='path to log directory')
    # -------- Huaizu: we currently don't need this parameter as train/val/test split has been done -------
    # p.add_argument('--val-ratio', default=0.05, type=float,
    #                help='ratio for validation split')
    # p.add_argument('--maxqlen', default=-1, type=int, help='max question length')
    # p.add_argument('--maxalen', default=-1, type=int, help='max answer length')
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
    p.add_argument('--im_vqa_file', required=True, type=str,
                   help='results file of pure image vqa')
    p.add_argument('--im_embed_file', required=True, type=str,
                   help='file of pre-computed image embeddings')

    config = p.parse_args()

    main(config)
