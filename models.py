from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.merge import add
from keras import regularizers
import keras.backend as K

import numpy as np


def fc_model(input_shape, layer_widths=[1024, 1024, 1024],
             activations=['tanh', 'tanh', 'tanh']):
    '''A simple feed-forward network.

    # Arguments
        input_shape: A network input shape. Expected a tuple of one integer.
        layer_widths: An iterable of integers which are the width of the
            corresponding network hidden dense layer.
        activations: An iterable of activations for each model hidden layer.

    # Returns
        Generated model.
    '''
    assert len(layer_widths) == len(activations)

    inp = Input(shape=input_shape, dtype=np.float32)
    layer = inp

    for w, activation in zip(layer_widths, activations):
        layer = Dense(w, activation='tanh',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01))(layer)

    layer = Dense(1, activation='relu')(layer)

    model = Model(inputs=[inp], outputs=[layer])
    model.compile(loss='mse', optimizer='rmsprop')

    return model

def cnn_model(input_shape, num_filters, num_blocks=4):
    '''A simple CNN network.

    # Arguments
        input_shape: A tuple of three dimensions. Power of two for spatial
            dimensions are preferred.
        num_filters: An iterable of the number of filters for each convolutional
            layers block (those convolutional layer which are before next max
            pooling layer).
        num_blocks: A number of convolution layers to be stacked before max
            pooling layer is placed.
    
    # Returns
        Generated model.
    '''
    order = int(np.log2(input_shape[0]))

    assert input_shape[0] == input_shape[1]
    assert len(num_filters) == order

    inp = Input(input_shape, dtype=np.float32)
    layer = inp

    for idx, filters in enumerate(num_filters):
        assert K.int_shape(layer)[1] == 2 ** (order - idx)
        for _ in range(num_blocks):
            layer = BatchNormalization()(layer)
            layer = Conv2D(filters=filters, kernel_size=3, padding='same',
                           kernel_regularizer=regularizers.l2(0.01),
                           bias_regularizer=regularizers.l2(0.01))(layer)
            layer = Activation('relu')(layer)

        layer = MaxPooling2D()(layer)

    assert K.int_shape(layer)[1] == 1

    layer = Conv2D(filters=num_filters[-1], kernel_size=1, padding='valid',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(layer)
    layer = Activation('relu')(layer)

    layer = Conv2D(filters=1, kernel_size=1, padding='valid',
                   kernel_regularizer=regularizers.l2(0.01),
                   bias_regularizer=regularizers.l2(0.01))(layer)
    layer = Activation('relu')(layer)

    layer = GlobalAveragePooling2D()(layer)

    model = Model(inputs=[inp], outputs=[layer])
    model.compile(loss='mse', optimizer='rmsprop')

    return model
