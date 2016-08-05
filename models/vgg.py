import numpy as np
import scipy as sp
import tensorflow as tf
import scipy.io


class Vgg19(object):
    LAYER_NAMES = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    STYLE_LAYER_NAMES = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
    CONTENT_LAYER_NAME = 'relu4_2'

    def __init__(self, path=None):
        self.model = scipy.io.loadmat(path or './pretrained/imagenet-vgg-verydeep-19.mat')
        self.pixel_mean = self.model['meta'][0]['normalization'][0][0][0][2][0][0]
        self.layers = self.model['layers'][0]

    def conv(self, inputs, index, name):
        weights, bias = self.layers[index][0][0][2][0]
        weights = np.transpose(weights, (1, 0, 2, 3))  # transposing width and height
        outputs = tf.nn.conv2d(
            inputs, filter=weights, strides=[1, 1, 1, 1], padding='SAME', name=name)
        return tf.nn.bias_add(outputs, bias.flatten())

    def pool(self, inputs, name):
        return tf.nn.max_pool(
            inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def build(self, placeholder):
        inputs = tf.convert_to_tensor(placeholder)
        net = {}
        for i, name in enumerate(self.LAYER_NAMES):
            layer_op = name[:4]
            if layer_op == 'conv':
                inputs = self.conv(inputs, i, name)
            elif layer_op == 'pool':
                inputs = self.pool(inputs, name)
            elif layer_op == 'relu':
                inputs = tf.nn.relu(inputs, name)
            net[name] = inputs
        return net



