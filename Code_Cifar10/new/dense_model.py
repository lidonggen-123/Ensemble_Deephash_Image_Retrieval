import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import config as c
import time as time
import tensorflow.keras.preprocessing.image as image
import matplotlib.pyplot as plt
from tensorflow.keras import Model, models, Input, regularizers
import os

def dense_block(x, blocks, name, growth_rate = 32):
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5,name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    filter = x.shape[3]
    x = layers.Conv2D(int(filter*reduction), 1,use_bias=False,name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(2 * growth_rate, 1,use_bias=False, name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3 ,padding='same',use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate( name=name + '_concat')([x, x1])
    return x

def my_densenet():
    inputs = keras.Input(shape=(32, 32, 3), name='img')
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    blocks = [4,8,16]
    x = dense_block(x, blocks[0], name='conv1',growth_rate =32)
    x = transition_block(x, 0.5, name='pool1')
    x = dense_block(x, blocks[1], name='conv2',growth_rate =32)
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[2], name='conv3',growth_rate =32)
    x = transition_block(x, 0.5, name='pool3')
    x = layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    hash_value = tf.keras.layers.Dense(units=c.hash_bit, activation=None, kernel_regularizer=regularizers.l2(0.01))(x)
    hash_value = tf.keras.layers.BatchNormalization()(hash_value)
    hash_value = tf.keras.layers.Activation("sigmoid")(hash_value)
    prediction = tf.keras.layers.Dense(units=c.num_class, activation="softmax")(hash_value)
    model = keras.Model(inputs=inputs, outputs=[prediction, hash_value])
    return model