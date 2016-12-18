from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def image_cnn():
	'''
	BUild a convolutional neural network (CNN) to get an embedding of an image.
	'''
	pass

def get_image_embedding():
	pass

def load_image_embedding(data_dir, image_id):
	'''
	Load 4096-dimensional image embedding using a VGG16 CNN pre-trained on ImageNet.
	'''
	file_name = os.path.join(data_dir, image_name)
	data = np.load(file_name)
	im_embed = data['im_embed']
	data.close()

	return im_embed