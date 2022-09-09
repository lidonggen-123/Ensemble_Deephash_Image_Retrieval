#!/usr/bin/python?
# -*- coding: utf-8 -*-import tensorflow as tf
import tensorflow as tf
import numpy as np
import pickle as p
from tqdm import tqdm
import os
import cv2
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Input
import datetime
from Mynet import *
weight_decay = 1e-4

# training config
batch_size = 128
train_num = 50000
iterations_per_epoch = int(train_num / batch_size)
learning_rate = [0.1, 0.01, 0.001]
boundaries = [80 * iterations_per_epoch, 120 * iterations_per_epoch]
epoch_num = 200

# test config
test_batch_size = 128
test_num = 10000
test_iterations = int(test_num / test_batch_size)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time + "-Resnet-cifar10"
summary_writer = tf.summary.create_file_writer(log_dir)

def hash_loss_fn(hash_input):
    loss1 = -1 * tf.reduce_mean(tf.square(hash_input - 0.5)) + 0.25  
    loss2 = tf.reduce_mean(tf.square(tf.reduce_mean(hash_input, axis=1) - 0.5))
    return loss1 + loss2


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR(Foldername):
    train_data = np.zeros([50000, 32, 32, 3], dtype=np.float32)
    train_label = np.zeros([50000, 10], dtype=np.float32)
    test_data = np.zeros([10000, 32, 32, 3], dtype=np.float32)
    test_label = np.zeros([10000, 10], dtype=np.float32)

    for sample in range(5):
        X, Y = load_CIFAR_batch(Foldername + "/data_batch_" + str(sample + 1))

        for i in range(3):
            train_data[10000 * sample:10000 * (sample + 1), :, :, i] = X[:, i, :, :]
        for i in range(10000):
            train_label[i + 10000 * sample][Y[i]] = 1

    X, Y = load_CIFAR_batch(Foldername + "/test_batch")
    for i in range(3):
        test_data[:, :, :, i] = X[:, i, :, :]
    for i in range(10000):
        test_label[i][Y[i]] = 1

    return train_data, train_label, test_data, test_label


def color_normalize(train_images, test_images):
    mean = [np.mean(train_images[:, :, :, i]) for i in range(3)]  # [125.307, 122.95, 113.865]
    std = [np.std(train_images[:, :, :, i]) for i in range(3)]  # [62.9932, 62.0887, 66.7048]
    for i in range(3):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    return train_images, test_images


def images_augment(images):
    output = []
    for img in images:
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        if np.random.randint(0, 2):
            img = cv2.flip(img, 1)
        output.append(img[x: x + 32, y:y + 32, :])
    return np.ascontiguousarray(output, dtype=np.float32)


def cross_entropy(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cross_entropy)


def l2_loss(model, weights=weight_decay):
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
        prediction, hash_value = model(x, training=True)  
        hash_loss = hash_loss_fn(hash_value)  
        ce = cross_entropy(y, prediction)
        l2 = l2_loss(model)
        loss = ce + l2 + hash_loss  
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return ce, hash_loss, prediction


@tf.function
def test_step(model, x, y):
    prediction, _ = model(x, training=False) 
    ce = cross_entropy(y, prediction)
    return ce, prediction


def train(model, optimizer, images, labels):
    sum_loss = 0
    sum_hash_loss = 0
    sum_accuracy = 0

    # random shuffle
    seed = np.random.randint(0, 65536)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    for i in tqdm(range(iterations_per_epoch)):
        x = images[i * batch_size: (i + 1) * batch_size, :, :, :]
        y = labels[i * batch_size: (i + 1) * batch_size, :]
        x = images_augment(x)

        loss, hash_loss, prediction = train_step(model, optimizer, x, y)
        sum_hash_loss += hash_loss
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('ce_loss:%f, l2_loss:%f,hash_loss:%f ,accuracy:%f' %
          (sum_loss / iterations_per_epoch, l2_loss(model), sum_hash_loss / iterations_per_epoch,
           sum_accuracy / iterations_per_epoch))
    return sum_loss / iterations_per_epoch, l2_loss(model), sum_hash_loss / iterations_per_epoch, sum_accuracy / iterations_per_epoch


def test(model, images, labels):
    sum_loss = 0
    sum_accuracy = 0

    for i in tqdm(range(test_iterations)):
        x = images[i * test_batch_size: (i + 1) * test_batch_size, :, :, :]
        y = labels[i * test_batch_size: (i + 1) * test_batch_size, :]
        x = tf.cast(x, tf.float32)  
        loss, prediction = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('test, loss:%f, accuracy:%f' %
          (sum_loss / test_iterations, sum_accuracy / test_iterations))
    return sum_accuracy / test_iterations


if __name__ == '__main__':
    # gpu config
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    import os

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(physical_devices)
    #tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # load data
    (train_images, train_labels, test_images, test_labels) = load_CIFAR('../dataset/cifar10')

    train_images, test_images = color_normalize(train_images, test_images)
    model = Resnet_Model()
    model.build((None,224,224,3))
    model.summary()
    #model = models.Model(img_input, output)
    
    # train
    #learning_rate_schedules = optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rate)
    #optimizer = optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)
    optimizer = optimizers.Adam()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)  
    for epoch in range(epoch_num):
        print('epoch %d' % epoch)
        ce_loss, l2_loss_, hash_loss, accuracy_train = train(model, optimizer, train_images, train_labels)
        with summary_writer.as_default():
            tf.summary.scalar('train-acc', float(accuracy_train), step=epoch)
        test_acc = test(model, test_images, test_labels)
        with summary_writer.as_default():
            tf.summary.scalar('test-acc', float(test_acc), step=epoch)
        if test_acc.numpy() > 0.80:
            name = "./ckeckpoints_hash_cifar10/48bit/Resnet/" + str(np.round(test_acc.numpy(), 3)) + '/model.ckpt'
            if not os.path.exists("./ckeckpoints_hash_cifar10/48bit/Resnet/" + str(np.round(test_acc.numpy(), 3))):
                os.makedirs("./ckeckpoints_hash_cifar10/48bit/Resnet/" + str(np.round(test_acc.numpy(), 3)))
            checkpoint.save(name)  