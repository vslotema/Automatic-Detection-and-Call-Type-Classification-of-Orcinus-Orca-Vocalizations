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
    AveragePooling2D,
    MaxPooling2D
)

from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from Augment import *

def _residual_block_enc(block_function, filters, repetitions, is_first_layer=False):

    def f(input):

        for i in range(repetitions):

            init_strides = (1, 1)
            if i == 0:
                init_strides = (2, 2)

            input = basic_block_enc(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)

        return input

    return f

def basic_block_enc(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    if is_first_block_of_first_layer:
        init_strides = (1,1)

    def f(input):

        out = Conv2D(filters,(3,3),strides=init_strides,padding="same",kernel_initializer='he_uniform',use_bias=False)(input)
        out =  BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(out)

        out = Activation('relu')(out)

        out =  Conv2D(filters,(3,3),strides=(1,1),padding="same",kernel_initializer='he_uniform',use_bias=False)(out)
        out = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(out)

        residual = _shortcut_enc(input,filters,strides=init_strides)

        out = add([residual,out])
        out = Activation("relu")(out)


        return out

    return f

def _shortcut_enc(input, filters,strides=(1,1)):

    stride_mean = sum(strides)/len(strides)
    shortcut = input

    if stride_mean > 1:

        shortcut = Conv2D(filters,(1,1),strides=strides,kernel_initializer='he_uniform',use_bias=False)(shortcut)
        shortcut = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(shortcut)

    return shortcut
############ Decoder

def _residual_block_dec(block_function, filters, repetitions, is_first_layer=False):

    def f(input):

        for i in range(repetitions):

            init_strides = (1, 1)
            if i == 1 :
                init_strides = (2, 2)
            print("i: {} init_strides: {}".format(i,init_strides))
            input = basic_block_dec(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 1))(input)

        return input

    return f

def basic_block_dec(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):


    def f(input):
        print("1 ", input.shape)
        out = Conv2D(filters,(3,3),strides=(1,1),padding="same",kernel_initializer='he_uniform',use_bias=False)(input)
        print("2 ", out.shape)
        out =  BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(out)
        print("3 ", out.shape)

        out = Activation('relu')(out)
        print("4 ", out.shape)
        if init_strides == (1,1):
            out = Conv2D(filters,(3,3),strides=init_strides,padding="same",kernel_initializer='he_uniform',use_bias=False)(out)
        else:
            out =  Conv2DTranspose(filters,(3,3),strides=init_strides,padding="same",kernel_initializer='he_uniform',use_bias=False)(out)
        print("5 ", out.shape)
        out = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(out)
        print("6 ", out.shape)

        residual = _shortcut_dec(input,filters,strides=init_strides)
        print("7 ", out.shape)

        out = add([residual,out])
        out = Activation("relu")(out)


        return out

    return f

def _shortcut_dec(input, filters,strides=(1,1)):

    stride_mean = sum(strides)/len(strides)
    shortcut = input
    print("filters: {} shortcut_filter:{} ".format(filters,shortcut.shape[3]))
    if stride_mean  > 1 or filters != shortcut.shape[3]:
        print("STRIDES ", strides)
        shortcut = Conv2DTranspose(filters,(1,1),strides=strides,kernel_initializer='he_uniform',use_bias=False)(shortcut)
        print("SHORTCUT")
        print("8 ", shortcut.shape)
        shortcut = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(shortcut)
        print("9 ", shortcut.shape)
        print("*******")

    return shortcut

def _decoder(inputshape):
   # latentInputs = Input(shape=(inputshape[1],inputshape[2],inputshape[3]))
    #print("lateninputs ", latentInputs)
    latentInputs = Input(shape=(512,))

    #x = Dense(np.prod(inputshape[1:]))(latentInputs)
    x = Reshape((inputshape[1], inputshape[2], inputshape[3]))(latentInputs)
    x = Conv2D(512, (1, 1), strides=1, padding="valid", use_bias=False, kernel_initializer='he_uniform')(
        x)
    print("***************", x.shape)
    x = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1, trainable=True)(x)
    x = Activation("relu")(x)

    x = _residual_block_dec(basic_block_dec,512,repetitions=2,is_first_layer=False)(x)
    x = _residual_block_dec(basic_block_dec,256,repetitions=2,is_first_layer=False)(x)
    x = _residual_block_dec(basic_block_dec,128,repetitions=2,is_first_layer=False)(x)
   # x = _residual_block_dec(basic_block_dec,64,repetitions=2,is_first_layer=False)(x)
    x = _residual_block_dec(basic_block_dec, 64, repetitions=2, is_first_layer=True)(x)
    x = Conv2DTranspose(1,(7,7),strides=(1,1),padding="same",use_bias=False,kernel_initializer='he_uniform')(x)
   # x = BatchNormalization(axis=3,epsilon=1e-05,momentum=0.1,trainable=True)(x)
    x = Activation("sigmoid")(x)

    return Model(latentInputs, x, name="decoder")

def _bottleneck(shape):
    input_bot = Input(shape=(shape[1], shape[2], shape[3]))
    x = Conv2D(4, (1, 1), strides=1, padding="valid", use_bias=False, kernel_initializer='he_uniform')(
        input_bot)
    x = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1, trainable=True)(x)
    x = Activation("relu")(x)
    shape = K.int_shape(x)
    x = Flatten()(x)
    #x = Dense(512)(x)

    bottleneck = Model(inputs=input_bot, outputs=x, name="bottleneck")
    return bottleneck,shape

class ResnetBuilder(object):

    @staticmethod
    def build(input_shape, block_fn, repetitions):

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
        #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(relu)

        block = relu
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block_enc(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        #block_shape = K.int_shape(block)
        #pool2 = AveragePooling2D(pool_size=(block_shape[1],block_shape[2]),
        #                         strides=(1, 1))(block)

        #flatten1 = Flatten()(pool2)
        #encoder = Dense(num_outputs)(flatten1)
        #print("encoder shape ", encoder.shape)
        encoder = block
        shape = K.int_shape(encoder)
        encoder = Model(inputs=input, outputs=encoder,name="encoder")

        ##BOTTLENECK
        bottleneck,shape = _bottleneck(shape)

        decoder = _decoder(shape)
        print("reached decoder")

        autoencoder = Model(inputs=input, outputs=decoder(bottleneck(encoder(input))),name="autoencoder")
        return (encoder,decoder,autoencoder)



    @staticmethod
    def build_resnet_18(input_shape):
        return ResnetBuilder.build(input_shape, basic_block_enc, [2, 2, 2, 2])
