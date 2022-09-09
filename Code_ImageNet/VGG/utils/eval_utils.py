import tensorflow as tf
import config as c


def cross_entropy_batch(y_true, y_pred, label_smoothing=0.0):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy


def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    acc = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return acc


def correct_num_batch(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num


def l2_loss(model, weights=c.weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def binary_loss(h):
    loss = -1 * tf.reduce_mean(tf.square(h - 0.5)) + 0.25  # 最大值为0.25
    return loss


def balancing_loss(h):
    mean = tf.reduce_mean(h, axis=1)  # 纵坐轴设置为1
    loss = tf.reduce_mean(tf.square(mean - 0.5))
    return loss


def binarize(h):
    """Convert to binary code vector"""
    # 1 (value >= 0.5)
    # 0 (value < 0.5)
    b = tf.math.floor(h + 0.5)
    return b
