#!/usr/bin/python?
# -*- coding: utf-8 -*-import tensorflow as tf
import numpy as np
import pickle as p
from tqdm import tqdm
import os
import cv2
from tensorflow.keras import models, optimizers, Sequential
from tensorflow.keras.layers import Input
from Mynet import Resnet_Model, DenseNet_Model, InceptionV3_Model, VGG19Net
from load_data import train_iterator, test_iterator
import tensorflow as tf
import math
import config as c


def getBinaryTensor(imgTensor, boundary=0.5):
    one = tf.ones_like(imgTensor)
    zero = tf.zeros_like(imgTensor)
    return tf.where(imgTensor > boundary, one, zero)


def hash_loss_fn(hash_input):
    loss1 = -1 * tf.reduce_mean(tf.square(hash_input - 0.5)) + 0.25  # 最大值为0.25
    loss2 = tf.reduce_mean(tf.square(tf.reduce_mean(hash_input, axis=1) - 0.5))
    return loss1 + loss2


def cross_entropy(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cross_entropy)


def l2_loss(model, weights=c.weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy


@tf.function
def test_step(model, x, y):
    prediction, hash_value = model(x, training=False)  # 修改
    ce = cross_entropy(y, prediction)
    return ce, prediction, hash_value


def test(model, test_iter):
    sum_loss = 0
    sum_accuracy = 0
    count = 0
    test_data_iterator, images = test_iterator("oxford-102-flowers/train.txt", c.num_class, c.batch_size)
    filewriter = open('hashcode/flower_12bit/hash_code_train_Desnet.txt', 'w')
    for i in tqdm(range(test_iter)):
        x, y = test_data_iterator.next()
        x = tf.cast(x, tf.float32)  # 将测试集中的图像编码成float32
        loss, prediction, hash_value = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)
        hash_value = getBinaryTensor(hash_value)
        for j in range(hash_value.shape[0]):
            filewriter.writelines(str(images[count]) + " " + str(hash_value[j].numpy()) + " " + str(np.argmax(y[j])) + "\n")
            count += 1
    print('test, loss:%f, accuracy:%f' %
          (sum_loss / c.iterations_per_epoch, sum_accuracy / c.iterations_per_epoch))
    return sum_accuracy / c.iterations_per_epoch


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    # get model
    #img_input = Input(shape=(224, 224, 3))
    #net = VGG19Net()
    #net.build((None,224,224,3))
    #output = net(img_input)
    #model = models.Model(img_input, output)

    model = DenseNet_Model()
    model.build((None, 32, 32, 3))
    
    checkpoint_save_path = "checkpoints_hash/flower_12bit/Desnet_400/0.93/"
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    print('-------------load the model-----------------')
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_save_path))
    print("loaded the model " + checkpoint_save_path + "model.ckpt-54")
    test_acc = test(model, c.iterations_per_epoch)
