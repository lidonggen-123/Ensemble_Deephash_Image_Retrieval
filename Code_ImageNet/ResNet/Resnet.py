#!/usr/bin/python?
# -*- coding: utf-8 -*-import tensorflow as tf
import numpy as np
import pickle as p
from tqdm import tqdm
import os
import cv2
from tensorflow.keras import models, optimizers, Sequential
from tensorflow.keras.layers import Input
from Mynet import VGG19Net, Resnet_Model
# from load_data import train_iterator, test_iterator
from utils.data_utils import train_iterator, test_iterator
import tensorflow as tf
import config as c
import datetime
 

class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

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
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        prediction, hash_value = model(x, training=True)  # 修改
        hash_loss = hash_loss_fn(hash_value)  # 修改
        ce = cross_entropy(y, prediction)
        l2 = l2_loss(model)
        loss = ce + l2 + hash_loss  # 修改
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return ce, hash_loss, prediction


@tf.function
def test_step(model, x, y):
    prediction, _ = model(x, training=False)  # 修改
    ce = cross_entropy(y, prediction)
    return ce, prediction


def train(model, optimizer, train_iterator, log_file):
    sum_loss = 0
    sum_hash_loss = 0
    sum_accuracy = 0

    for i in tqdm(range(c.iterations_per_epoch)):
        x, y = train_iterator.next()
        loss, hash_loss, prediction = train_step(model, optimizer, x, y)
        sum_hash_loss += hash_loss
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('ce_loss:%f, l2_loss:%f,hash_loss:%f ,accuracy:%f' %
          (sum_loss / c.iterations_per_epoch, l2_loss(model), sum_hash_loss / c.iterations_per_epoch,
           sum_accuracy / c.iterations_per_epoch))
    log_file.write('train: ce loss: {:.4f}, l2 loss: {:.4f}, hash_loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_loss / c.iterations_per_epoch, l2_loss(model), 
    sum_hash_loss / c.iterations_per_epoch, sum_accuracy / c.iterations_per_epoch ))


def test(model, test_iter, log_file):
    sum_loss = 0
    sum_accuracy = 0
    test_data_iterator = test_iterator()

    for i in tqdm(range(test_iter)):
        x, y = test_data_iterator.next()
        x = tf.cast(x, tf.float32)  # 将测试集中的图像编码成float32
        loss, prediction = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)
    print('test, loss:%f, accuracy:%f' %
          (sum_loss / c.test_iterations, sum_accuracy / c.test_iterations))
    log_file.write('test: loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_loss / c.test_iterations, sum_accuracy / c.test_iterations))
    return sum_accuracy / c.test_iterations


if __name__ == '__main__':
    # gpu config
    #gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    #for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu, True)
    import os

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    #print(physical_devices)

    #tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # show
    model = Resnet_Model()
    model.build((None,224,224,3))
    model.summary()

    # train-cos
    learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=c.initial_learning_rate,
                                                decay_steps=c.epoch_num * c.iterations_per_epoch - c.warm_iterations,
                                                alpha=c.minimum_learning_rate,
                                                warm_up_step=c.warm_iterations)
    optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    checkpoint.restore(tf.train.latest_checkpoint('/root/shy/train_imagenet_res/checkpoints_hash_imagenet/512bit_imagenet/Resnet/0.632'))
    # train
    #learning_rate_schedules = optimizers.schedules.PiecewiseConstantDecay(c.boundaries, c.learning_rate)
    #optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)
    
    # optimizer = optimizers.Adam(learning_rate=learning_rate_schedules)
    # optimizer = optimizers.Adam()
    #checkpoint = tf.train.Checkpoint(myAwesomeModel=model)

    train_data_iterator = train_iterator()
    for epoch in range(c.epoch_num):
        print(epoch)
        with open(c.log_file, 'a') as f:
            f.write('epoch:{}\n'.format(epoch))
            train(model, optimizer, train_data_iterator, f)
            test_acc = test(model, c.test_iterations, f)
        if test_acc > 0:
            name = "./checkpoints_hash_imagenet/512bit_imagenet_418/Resnet/" + str(np.round(test_acc.numpy(), 3)) + '/model.ckpt'
            if not os.path.exists("./checkpoints_hash_imagenet/512bit_imagenet_418/Resnet/" + str(np.round(test_acc.numpy(), 3))):
                os.makedirs("./checkpoints_hash_imagenet/512bit_imagenet_418/Resnet/" + str(np.round(test_acc.numpy(), 3)))
            checkpoint.save(name)
