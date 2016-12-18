from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

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
      visual_concepts     a list where each entry is a dictionary containing
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

  with open(ground_annot_file, 'r') as f:
    ground_data = json.load(f)['boxes']

  # build a map from qa_id to image id, to group visual concepts in an image
  qa_id_image_id = {}
  for split in ['train', 'val', 'test']:
    for qa in qa_data[split]:
      qa_id_image_id[qa['qa_id']] = qa['image_id']

  image_concepts = {}
  for gd in ground_data:
    image_id = qa_id_image_id[gd['qa_id']]
    if image_id in image_concepts:
      image_concepts[image_id].append(gd)
    else:
      image_concepts[image_id] = [gd];

  for split in ['train', 'val', 'test']:
    for idx, qa in enumerate(qa_data[split]):
      image_id = qa['image_id']
      if image_id in image_concepts:
        visual_concepts = image_concepts[image_id]
      else:
        visual_concetps = []
      # qa.update({'visual_concepts': image_concepts[image_id]})
      qa['visual_concepts'] = visual_concepts
      qa_data[split][idx] = qa

  return qa_data

def append_image_vqa_results(vqa_data, image_vqa_results_file):
  '''
  Augment the visual question answering dataset with predicted answers from a pure image vqa model.

  Input 
    vqa_data                  data returned by the read_visual7w_dataset function
    image_vqa_results_file    file path containing the image vqa predictions

  Output
    Similar vqa_data, but each of the dictionary contains more data as follows
      logp      a list of float values, where each entry is the average negative log likelihood 
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
      vqa_data[split][idx]['logp'] = p['logp']
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
    image_id_embed = json.load(f)

  for split in ['train', 'val', 'test']:
    for idx in xrange(len(vqa_data[split])):
      image_id = vqa_data[split][idx]['image_id']
      vqa_data[split][idx]['im_embed'] = image_id_embed[image_id]

def preprocess_raw_data(vqa_data):
  pass
  
def vqa_data_iterator(vqa_data, batch_size):
	pass