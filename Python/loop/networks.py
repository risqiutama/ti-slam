"""
Network definitions
"""

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from pylab import *
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.datasets import mnist

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense, Reshape
from keras.initializers import glorot_uniform,he_uniform

from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K

# from keras.backend import l2_normalize, expand_dims, variable, constant
from keras.utils import plot_model, normalize
import tensorflow as tf
import numpy as np
from keras import initializers, layers
from keras.layers import Conv2D


def base_network(input_size, descriptor_size, trainable=False):
    '''
    Define the base network model
    :param input_size: shape of input images
    :param descriptor_size: embedding size used to encode input images
    :return: network model
    '''
    base_model = ResNet50(input_shape=input_size, weights='imagenet', include_top=False)
    # base_model.trainable = trainable

    x = GlobalMaxPooling2D(name='global_max_1')(base_model.get_layer('activation_49').output)
    # x = GlobalMaxPooling2D(name='global_max_1')(base_model.get_layer('block4_pool').output)
    x = Dense(descriptor_size * 4, kernel_regularizer=l2(1e-3), activation='relu', kernel_initializer='he_uniform', name='dense_descriptor_1')(x)
    x = Dense(descriptor_size * 2, kernel_regularizer=l2(1e-3), activation='relu', kernel_initializer='he_uniform', name='dense_descriptor_2')(x)
    descriptor = Dense(descriptor_size, kernel_regularizer=l2(1e-3), kernel_initializer='he_uniform', name='dense_descriptor_3')(x)
    norm_descriptor = Lambda(lambda x: K.l2_normalize(x, axis=-1))(descriptor)
    network = Model(inputs=[base_model.input], outputs=[norm_descriptor])
    # return norm_descriptor
    network.summary()

    for layer in base_model.layers:
        layer.trainable = trainable

    return network


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square( anchor -positive), axis=-1)
        n_dist = K.sum(K.square( anchor -negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

class TripletLossLayerKL(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayerKL, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = keras.losses.kullback_leibler_divergence(anchor, positive)
        n_dist = keras.losses.kullback_leibler_divergence(anchor, negative)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def build_neural_embedding(input_shape, base_network, margin_loss):
    '''
    :param input_shape: shape of input images
    :param network: descriptor/embedding size
    :param margin_loss: margin in triplet loss
    :return: network definition for training
    '''
    # Define input tensor
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')

    anchor_merge = Concatenate(axis=-1)([anchor_input, anchor_input, anchor_input])
    pos_merge = Concatenate(axis=-1)([positive_input, positive_input, positive_input])
    neg_merge = Concatenate(axis=-1)([negative_input, negative_input, negative_input])

    net_anchor = base_network(anchor_merge)
    net_positive = base_network(pos_merge)
    net_negative = base_network(neg_merge)

    # TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin_loss, name='triplet_loss_layer')([net_anchor, net_positive, net_negative])

    # Connect the inputs with the outputs
    network_training = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss_layer)

    return network_training