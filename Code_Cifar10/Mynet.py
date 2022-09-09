import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model,models,Input,regularizers
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3, DenseNet121, VGG16
import config as c

class VGG19Net(Model):
    def __init__(self):
        super(VGG19Net, self).__init__()
        self.base_model = VGG19(weights='imagenet', include_top=False, pooling='max')
        # self.base_model.trainable = False
        #for i in range(len(self.base_model.layers) - 3):
        #    self.base_model.layers[i].trainable = False
        self.d1 = Dense(units=c.hash_bit, activation=None)
        self.bn1 = BatchNormalization()
        self.ac1 = tf.keras.layers.Activation("sigmoid")
        self.d2 = Dense(units=c.num_class, activation=tf.nn.softmax)

    def call(self, x):
        x = self.base_model(x)
        hash_code = self.d1(x)
        hash_code = self.bn1(hash_code)
        hash_code = self.ac1(hash_code)
        prediction = self.d2(hash_code)
        return prediction, hash_code


class VGG16Net(Model):
    def __init__(self):
        super(VGG16Net, self).__init__()
        self.base_model = VGG16(weights='imagenet', include_top=False, pooling='max')
        # self.base_model.trainable = False
        for i in range(len(self.base_model.layers) - 2):  # print(len(model.layers))=23
            self.base_model.layers[i].trainable = False
        self.d1 = Dense(units=c.hash_bit, activation=tf.nn.sigmoid)
        self.d2 = Dense(units=c.num_class, activation=tf.nn.softmax)

    def call(self, x):
        x = self.base_model(x)
        hash_code = self.d1(x)
        prediction = self.d2(hash_code)
        return prediction, hash_code


def Resnet_Model():
    base_model = ResNet50(weights='imagenet', include_top=False, layers=tf.keras.layers, input_shape=(224, 224, 3))
    print(len(base_model.layers))  # 175
    #for layer in base_model.layers[:170]:
    #    layer.trainable = False
    #for layer in base_model.layers[170:]:
    #    layer.trainable = True
        
    #for i in range(len(base_model.layers) - 4):
    #    base_model.layers[i].trainable = False
        
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    hash_value = tf.keras.layers.Dense(units=c.hash_bit, activation=None, kernel_regularizer=regularizers.l2(0.01))(x)
    #hash_value = tf.keras.layers.Dense(units=c.hash_bit, activation=None)(x)
    hash_value = tf.keras.layers.BatchNormalization()(hash_value)
    hash_value = tf.keras.layers.Activation("sigmoid")(hash_value)
    prediction = tf.keras.layers.Dense(units=c.num_class, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(hash_value)
    #prediction = tf.keras.layers.Dense(units=c.num_class, activation="softmax")(hash_value)
    model = models.Model(inputs=base_model.input, outputs=[prediction, hash_value])
    return model


def DenseNet_Model():
    base_model = DenseNet121(weights='imagenet', include_top=False, layers=tf.keras.layers, input_shape=(224, 224, 3))
    print(len(base_model.layers))  # 427  
    for layer in base_model.layers[:415]:
        layer.trainable = False
    for layer in base_model.layers[415:]:
        layer.trainable = True
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    hash_value = tf.keras.layers.Dense(units=c.hash_bit, activation=None)(x)
    hash_value = tf.keras.layers.BatchNormalization()(hash_value)
    hash_value = tf.keras.layers.Activation("sigmoid")(hash_value)
    prediction = tf.keras.layers.Dense(units=c.num_class, activation="softmax")(hash_value)
    model = models.Model(inputs=base_model.input, outputs=[prediction, hash_value])
    return model


# def InceptionV3_Model():
#     base_model = InceptionV3(weights='imagenet', include_top=False, layers=tf.keras.layers, input_shape=(224, 224, 3))
#     print(len(base_model.layers))  # 311
#     for layer in base_model.layers[:306]:
#         layer.trainable = False
#     for layer in base_model.layers[306:]:
#         layer.trainable = True
#     x = base_model.output
#     x = tf.keras.layers.GlobalAveragePooling2D(name='average_pool')(x)
#     x = tf.keras.layers.Flatten(name='flatten')(x)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     hash_value = tf.keras.layers.Dense(units=48, activation=None)(x)
#     hash_value = tf.keras.layers.BatchNormalization()(hash_value,training=False)
#     hash_value = tf.keras.layers.Activation("sigmoid")(hash_value)
#     prediction = tf.keras.layers.Dense(units=c.num_class, activation="softmax")(hash_value)
#     model = models.Model(inputs=base_model.input, outputs=[prediction, hash_value])
#     return model

def InceptionV3_Model():
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    #冻结前面的层，训练最后20层
    for layers in base_model.layers[:-5]:
        layers.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = tf.keras.layers.Flatten(name='flatten')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    hash_value = tf.keras.layers.Dense(units=c.hash_bit, activation=None)(x)
    hash_value = tf.keras.layers.BatchNormalization()(hash_value)
    #hash_value = tf.keras.layers.BatchNormalization()(hash_value,training=False)
    hash_value = tf.keras.layers.Activation("sigmoid")(hash_value)
    prediction = tf.keras.layers.Dense(units=c.num_class, activation="softmax")(hash_value)
    model = models.Model(inputs=base_model.input, outputs=[prediction, hash_value])
    return model