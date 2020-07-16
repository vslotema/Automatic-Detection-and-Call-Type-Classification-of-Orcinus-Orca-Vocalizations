from __future__ import division

import numpy as np
import math
import six
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    AveragePooling2D
)

from keras.layers import Conv2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K



def _residual_block(block_function, filters, repetitions, is_first_layer=False):

    def f(input):

        for i in range(repetitions):

            init_strides = (1, 1)
            if i == 0:
                init_strides = (2, 2)

            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f

def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):

    if is_first_block_of_first_layer:
        init_strides = (1,1)

    def f(input):

        out = Conv2D(filters,(3,3),strides=init_strides,padding="same",kernel_initializer='he_uniform',use_bias=False)(input)
        out =  BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(out)

        out = Activation('relu')(out)

        out =  Conv2D(filters,(3,3),strides=(1,1),padding="same",kernel_initializer='he_uniform',use_bias=False)(out)
        out = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(out)

        residual = _shortcut(input,filters,strides=init_strides)

        out = add([residual,out])
        out = Activation("relu")(out)


        return out

    return f

def _shortcut(input, filters,strides=(1,1)):

    stride_mean = sum(strides)/len(strides)
    shortcut = input

    if stride_mean > 1:
        shortcut = Conv2D(filters,(1,1),strides=strides,padding="same",kernel_initializer='he_uniform',use_bias=False)(shortcut)
        shortcut = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(shortcut)
    return shortcut

class ResnetBuilder(object):

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):

        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_cols, nb_rows,nb_channels)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple  (nb_cols, nb_rows, nb_channels)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_first':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input = Input(shape=input_shape)
        conv1 = Conv2D(64,(7,7),strides=(2,2),padding="same",use_bias=False,kernel_initializer='he_uniform')(input)
        bn1 = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(conv1)
        relu = Activation("relu")(bn1)
       # pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = relu
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2


        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[1],block_shape[2]),
                                 strides=(1, 1))(block)

        flatten1 = Flatten()(pool2)

        dense = Dense(units=num_outputs,
                     activation="softmax", use_bias=True,kernel_initializer="he_normal")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])
