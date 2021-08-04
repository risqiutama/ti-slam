"""
Network definitions
"""

import numpy as np
np.random.seed(0)
# import matplotlib.pyplot as plt
# from pylab import *
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model, load_model
from keras.datasets import mnist

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense, Reshape
from keras.initializers import glorot_uniform,he_uniform
from keras.layers import TimeDistributed, LeakyReLU, concatenate, merge, GlobalAveragePooling2D, AveragePooling2D, LSTM, Dropout, Conv2D, Multiply
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
import mdn
# from keras.backend import l2_normalize, expand_dims, variable, constant
from keras.utils import plot_model, normalize
import tensorflow as tf
import numpy as np
from keras import initializers, layers
from os.path import join

def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta

  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))

def FlowNet_module(input, dup=False, hallu_name='_rgb'):
    if not dup:
        net = TimeDistributed(Conv2D(64, 7, strides=(2, 2), padding='same'), name='conv1')(input)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU1')(net)
        net = TimeDistributed(Conv2D(128, 5, strides=(2, 2), padding='same'), name='conv2')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU2')(net)
        net = TimeDistributed(Conv2D(256, 5, strides=(2, 2), padding='same'), name='conv3')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU3')(net)
        net = TimeDistributed(Conv2D(256, 3, strides=(1, 1), padding='same'), name='conv3_1')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU4')(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(2, 2), padding='same'), name='conv4')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU5')(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(1, 1), padding='same'), name='conv4_1')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU6')(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(2, 2), padding='same'), name='conv5')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU7')(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(1, 1), padding='same'), name='conv5_1')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU8')(net)
        net = TimeDistributed(Conv2D(1024, 3, strides=(2, 2), padding='same'), name='conv6')(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU9')(net)
    else:
        net = TimeDistributed(Conv2D(64, 7, strides=(2, 2), padding='same'), name='conv1' + hallu_name)(input)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU1' + hallu_name)(net)
        net = TimeDistributed(Conv2D(128, 5, strides=(2, 2), padding='same'), name='conv2' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU2' + hallu_name)(net)
        net = TimeDistributed(Conv2D(256, 5, strides=(2, 2), padding='same'), name='conv3' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU3' + hallu_name)(net)
        net = TimeDistributed(Conv2D(256, 3, strides=(1, 1), padding='same'), name='conv3_1' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU4' + hallu_name)(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(2, 2), padding='same'), name='conv4' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU5' + hallu_name)(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(1, 1), padding='same'), name='conv4_1' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU6' + hallu_name)(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(2, 2), padding='same'), name='conv5' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU7' + hallu_name)(net)
        net = TimeDistributed(Conv2D(512, 3, strides=(1, 1), padding='same'), name='conv5_1' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU8' + hallu_name)(net)
        net = TimeDistributed(Conv2D(1024, 3, strides=(2, 2), padding='same'), name='conv6' + hallu_name)(net)
        net = TimeDistributed(LeakyReLU(alpha=0.1), name='ReLU9' + hallu_name)(net)
    return net

# USE THIS ONE
def build_neural_odometry(cfg, input_shape = (1, 512, 640, 3), imu_length=10, istraining=True, isfirststage=1,
                                 base_model_name='deeptio_handheld_v2', n_mixture=10):
    # MDN settings
    OUTPUT_DIMS = 3
    N_MIXES = n_mixture

    image_1 = Input(shape=input_shape, name='image_1')
    image_2 = Input(shape=input_shape, name='image_2')

    image_merged = concatenate([image_1, image_2], axis=-1, name='image_merged')

    # --- thermal data
    net = FlowNet_module(image_merged)
    avg_pool_1 = TimeDistributed(AveragePooling2D(pool_size=(4, 5), strides=None, padding='valid', data_format=None),
                                 name='avg_pool_1')(net)
    avg_pool_2 = TimeDistributed(AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None),
                                 name='avg_pool_2')(avg_pool_1)
    conv_thermal_feature = TimeDistributed(Flatten(), name='flatten')(avg_pool_2)

    # --- hallucinated rgb data
    net = FlowNet_module(image_merged, dup=True, hallu_name='_rgb')
    avg_pool_1_rgb = TimeDistributed(AveragePooling2D(pool_size=(4, 5), strides=None, padding='valid', data_format=None),
                                    name='avg_pool_1_rgb')(net)
    avg_pool_2_rgb = TimeDistributed(AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid', data_format=None),
                                    name='avg_pool_2_rgb')(avg_pool_1_rgb)
    conv_hallucination_feature = TimeDistributed(Flatten(), name='flatten_rgb')(avg_pool_2_rgb)

    # IMU data
    imu_data = Input(shape=(imu_length, 6), name='imu_data')

    imu_states = 256
    imu_lstm_1 = LSTM(imu_states, return_sequences=True, name='imu_lstm_1')(imu_data) # 128, 256
    reshape_imu = Reshape((1, imu_length * imu_states))(imu_lstm_1)  # 2560, 5120, 10240

    # merge features
    merge_features = concatenate([conv_thermal_feature, conv_hallucination_feature, reshape_imu],
                           axis=-1, name='merge_features')

    # selective merge feature
    dense_selective_thermal = Dense(2048, activation='sigmoid', use_bias=False, name='dense_selective_thermal')(merge_features)
    dense_selective_hallucination = Dense(2048, activation='sigmoid', use_bias=False, name='dense_selective_hallucination')(merge_features)
    dense_selective_imu = Dense(imu_length * imu_states, activation='sigmoid', use_bias=False, name='dense_selective_imu')(merge_features)

    selective_thermal_features = Multiply()([conv_thermal_feature, dense_selective_thermal])
    selective_hallucination_features = Multiply()([conv_hallucination_feature, dense_selective_hallucination])
    selective_imu_features = Multiply()([reshape_imu, dense_selective_imu])

    selective_features = concatenate([selective_thermal_features, selective_hallucination_features, selective_imu_features],
                               axis=-1, name='selective_features')

    forward_lstm_1 = LSTM(512, dropout_W=0.25, return_sequences=True, name='forward_lstm_1')(selective_features)
    forward_lstm_2 = LSTM(512, return_sequences=True, name='forward_lstm_2')(forward_lstm_1)

    fc_position_1 = TimeDistributed(Dense(128, activation='relu'), name='fc_position_1')(forward_lstm_2)  # tanh
    mdn_trans = TimeDistributed(mdn.MDN(OUTPUT_DIMS, N_MIXES))(fc_position_1)

    fc_orientation_1 = TimeDistributed(Dense(128, activation='relu'), name='fc_orientation_1')(forward_lstm_2)  # tanh
    mdn_rot = TimeDistributed(mdn.MDN(OUTPUT_DIMS, N_MIXES))(fc_orientation_1)

    if istraining:
        model = Model(inputs=[image_1, image_2, imu_data], outputs=[mdn_trans, mdn_rot, conv_hallucination_feature])

        layers_names = ['conv1', 'conv2', 'conv3', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv5_1', 'conv6',
                        'conv1_rgb', 'conv2_rgb', 'conv3_rgb', 'conv3_1_rgb', 'conv4_rgb', 'conv4_1_rgb', 'conv5_rgb',
                        'conv5_1_rgb', 'conv6_rgb']
        hallucination_layers = ['conv1_rgb', 'conv2_rgb', 'conv3_rgb', 'conv3_1_rgb', 'conv4_rgb', 'conv4_1_rgb',
                                'conv5_rgb', 'conv5_1_rgb', 'conv6_rgb']
        thermal_layers = ['conv1', 'conv2', 'conv3', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv5_1', 'conv6']
        fc_regression_layers = ['fc_position_1', 'fc_position_2', 'fc_trans',
                                'fc_orientation_1', 'fc_orientation_2', 'fc_rot',
                                'time_distributed_1', 'time_distributed_2']

        if int(isfirststage) == 1:
            # FIRST STAGE of training, use flownet to initialize both thermal n hallucination network
            # but then train hallucination only (freeze thermal layers)
            CNN_model = load_model(join('./models', 'cnn.h5'))
            for layer in model.layers:
                if layer.name in layers_names:
                    if layer.name in hallucination_layers:
                        layer_name = layer.name
                        string_name = layer_name[:-4]  # name to call the model in flownet (cnn)
                        # print('Detected hallucination layer: ', string_name, '_rgb.')
                    else:
                        string_name = layer.name

                    weights, biases = CNN_model.get_layer(name=string_name).get_weights()
                    model.get_layer(name=layer.name).set_weights([weights, biases])
                    print('--', layer.name, ':', weights.shape, model.get_layer(name=layer.name).get_weights()[0].shape)

            # freezing thermal
            for layer in model.layers:
                if layer.name in thermal_layers:
                        print('Freeze layer :', str(layer.name))
                        layer.trainable = False

            optimizer = Adam(lr=cfg['lr_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        else:
            # load model with the hallucination part that has been trained
            print('Load the base model: ' + base_model_name)
            model_path = join('./models', base_model_name, 'best')
            model.load_weights(model_path, by_name=True)

            for layer in model.layers:
                if layer.name in layers_names:  # freeze thermal and hallucination
                    print('Freeze layer :', str(layer.name))
                    layer.trainable = False

            # Train regression only
            # for layer in model.layers:
            #     if layer.name in fc_regression_layers: # regression_layers, fc_regression_layers
            #         layer.trainable = True
            #         print('Trainable layer :', str(layer.name))
            #     else:
            #         layer.trainable = False

            optimizer = Adam(lr=cfg['lr_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)


        # configure learning process with compile()
        model.compile(optimizer=optimizer, loss={'time_distributed_1': mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES),
                                                 'time_distributed_2': mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES),
                                                 'flatten_rgb': huber_loss_mean},
                      loss_weights={'time_distributed_1': cfg['fc_trans'],
                                    'time_distributed_2': cfg['fc_rot'], 'flatten_rgb': cfg['flatten_rgb']})
    else:
        model = Model(inputs=[image_1, image_2, imu_data], outputs=[mdn_trans, mdn_rot])
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(optimizer=optimizer, loss={'time_distributed_1': mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                                                 'time_distributed_2': mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)},
                      loss_weights={'time_distributed_1': 1,
                                    'time_distributed_2': 1})
        for layer in model.layers[:]:
            layer.trainable = False
        # load weights
        model.load_weights(cfg)

    return model

# USE THIS ONE
def build_neural_loop_closure(cfg, input_shape=(1, 512, 640, 3), istraining=True, hallu_name = '_rgb'):
    '''
    This network structure follows DeepTIO, so we can initialize the weights with DeepTIO
    '''
    # MDN settings
    OUTPUT_DIMS = 3
    N_MIXES = 10

    image_1 = Input(shape=input_shape, name='image_1')
    image_2 = Input(shape=input_shape, name='image_2')

    image_merged = concatenate([image_1, image_2], axis=-1)

    # Thermal flow features extractor
    net = FlowNet_module(image_merged)
    avg_pool_1 = TimeDistributed(AveragePooling2D(pool_size=(4, 5), strides=None, padding='valid'),
                                 name='avg_pool_1')(net)
    avg_pool_2 = TimeDistributed(AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid'),
                                 name='avg_pool_2')(avg_pool_1)
    conv_thermal_feature = TimeDistributed(Flatten(), name='flatten')(avg_pool_2)

    # Hallucination flow features extractor
    net = FlowNet_module(image_merged, dup=True)
    avg_pool_1_hallucination = TimeDistributed(AveragePooling2D(pool_size=(4, 5), strides=None, padding='valid'),
                                 name='avg_pool_1'+hallu_name)(net)
    avg_pool_2_hallucination = TimeDistributed(AveragePooling2D(pool_size=(2, 1), strides=None, padding='valid'),
                                 name='avg_pool_2'+hallu_name)(avg_pool_1_hallucination)
    conv_hallucination_feature = TimeDistributed(Flatten(), name='flatten_hallu')(avg_pool_2_hallucination)

    merge_features = concatenate([conv_thermal_feature, conv_hallucination_feature], axis=-1, name='merge_features')

    # Selective features (thermal, hallucination)
    dense_selective_thermal = Dense(2048, activation='sigmoid', use_bias=False, name='dense_selective_thermal')(
        merge_features)  # 2048, 4096
    dense_selective_hallucination = Dense(2048, activation='sigmoid', use_bias=False,
                                          name='dense_selective_hallucination')(merge_features)

    selective_thermal_features = Multiply()([conv_thermal_feature, dense_selective_thermal])
    selective_hallucination_features = Multiply()([conv_hallucination_feature, dense_selective_hallucination])

    selective_features = concatenate([selective_thermal_features, selective_hallucination_features],
                               axis=-1, name='selective_features')

    # Pose Regressor
    forward_lstm_1 = LSTM(512, dropout=0.25, return_sequences=True, name='forward_lstm_1')(selective_features)
    forward_lstm_2 = LSTM(512, return_sequences=True, name='forward_lstm_2')(forward_lstm_1)

    fc_position_1 = TimeDistributed(Dense(128, activation='relu'), name='fc_position_1')(forward_lstm_2)
    mdn_trans = TimeDistributed(mdn.MDN(OUTPUT_DIMS, N_MIXES))(fc_position_1)

    fc_orientation_1 = TimeDistributed(Dense(128, activation='relu'), name='fc_orientation_1')(forward_lstm_2)
    mdn_rot = TimeDistributed(mdn.MDN(OUTPUT_DIMS, N_MIXES))(fc_orientation_1)

    if istraining:

        model = Model(inputs=[image_1, image_2], outputs=[mdn_trans, mdn_rot])

        layers_names = ['conv1', 'conv2', 'conv3', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv5_1', 'conv6',
                        'conv1' + hallu_name, 'conv2' + hallu_name, 'conv3' + hallu_name, 'conv3_1' + hallu_name,
                        'conv4' + hallu_name,
                        'conv4_1' + hallu_name, 'conv5' + hallu_name, 'conv5_1' + hallu_name, 'conv6' + hallu_name]

        regression_layers = ['forward_lstm_1', 'forward_lstm_2', 'fc_position_1', 'fc_position_2', 'fc_trans',
                             'fc_orientation_1', 'fc_orientation_2', 'fc_rot']

        fc_regression_layers = ['fc_position_1', 'fc_position_2', 'fc_trans',
                                'fc_orientation_1', 'fc_orientation_2', 'fc_rot',
                                'time_distributed_1', 'time_distributed_2']

        # Initialise from DeepTIO (handheld)
        model_path = join('./models/neural_loop_closure/', 'best')

        CNN_model = load_model(model_path,  custom_objects={'MDN': mdn.MDN,
                                                            'loss_func': mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES),
                                                            'huber_loss_mean': huber_loss_mean})

        # CNN_model = load_model(model_path, custom_objects={'huber_loss_mean': huber_loss_mean})

        for layer in model.layers:
            if layer.name in layers_names:
                weights, biases = CNN_model.get_layer(name=layer.name).get_weights()
                model.get_layer(name=layer.name).set_weights([weights, biases])
                print('--', layer.name, ':', weights.shape, model.get_layer(name=layer.name).get_weights()[0].shape)

        # Freeze layers
        for layer in model.layers:
            if layer.name in layers_names:
                    print('Freeze layer :', str(layer.name))
                    layer.trainable = False

        # for layer in model.layers:
        #     if layer.name in layers_names or layer.name in regression_layers:  # freeze thermal, hallucination, regression
        #             print('Freeze layer :', str(layer.name))
        #             layer.trainable = False

        # Train regression only
        # for layer in model.layers:
        #     if layer.name in fc_regression_layers: # regression_layers, fc_regression_layers
        #         layer.trainable = True
        #         print('Trainable layer: ', layer.name)
        #     else:
        #         layer.trainable = False

        # Configure learning process with compile()
        # rmsProp = RMSprop(lr=cfg['lr_rate'], rho=cfg['rho'],
        #                                    epsilon=float(cfg['epsilon']),
        #                                    decay=cfg['decay'])  # regressor: 0.001, ft cnn: 0.0001
        adam = Adam(lr=cfg['lr_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss={'time_distributed_1': mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES),
                                            'time_distributed_2': mdn.get_mixture_loss_func(OUTPUT_DIMS,N_MIXES)},
                      loss_weights={'time_distributed_1': cfg['fc_trans'],
                                    'time_distributed_2': cfg['fc_rot']})
    else:
        model = Model(inputs=[image_1, image_2], outputs=[mdn_trans, mdn_rot])
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        model.compile(optimizer=optimizer, loss={'time_distributed_1': mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES),
                                                 'time_distributed_2': mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)},
                      loss_weights={'time_distributed_1': 1,
                                    'time_distributed_2': 1})

        for layer in model.layers[:]:
            layer.trainable = False

        model.load_weights(cfg)

    return model