#!/usr/bin/python3
#coding=utf-8
#coding=gbk
import tensorflow as tf
import numpy as np
import cv2


def load_list(list_path):
    images = []
    labels = []
    print(list_path)
    with open(list_path, 'r') as f:
        for line in f:
            lines = line.replace('\n', '').split('\t')
            if len(lines) == 1:
                lines = line.replace('\n', '').split()
            # print(lines)
            # images.append(os.path.join(image_root_path, line[0]))
            images.append(lines[0])
            labels.append(int(lines[1]))
    return images, labels


def random_size(image, target_size=None):
    height, width, _ = np.shape(image)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width
    resize_shape = (int(width * size_ratio), int(height * size_ratio))
    return cv2.resize(image, resize_shape)


def normalize(image):
    # iamgenet褰掍竴鍖栦腑鐢ㄥ埌鐨勬暟鎹?
    mean = [103.939, 116.779, 123.68]
    std = [58.393, 57.12, 57.375]
    for i in range(3):
        image[..., i] = (image[..., i] - mean[i]) / std[i]
    return image


def random_crop(image):
    height, width, _ = np.shape(image)
    if height < width:
        num = width - height
        import random
        value = random.randint(0, num)
        value //= 2
        image = image[:, value:224 + value]
        image = cv2.resize(image, (224, 224))

    else:
        num = height - width
        import random
        value = random.randint(0, num)
        value //= 2
        image = image[value:224 + value, :]
        image = cv2.resize(image, (224, 224))
    return image

#现在我们的jpg文件进行解码，变成三维矩阵
def load_preprosess_image(path,label, category_num):
    #读取路径
    image = cv2.imread(path.numpy().decode()).astype(np.float32)
    #解码
    #image=tf.image.decode_jpeg(image,channels=3)#彩色图像为3个channel
    #将图像改变为同样的大小，利用裁剪或者扭曲,这里应用了扭曲
    image=tf.image.resize(image,[256,256])
    #随机裁剪图像
    image=tf.image.random_crop(image,[224,224,3])
    #随机上下翻转图像
    image=tf.image.random_flip_left_right(image)
    #随机上下翻转
    image=tf.image.random_flip_up_down(image)
    #随机改变图像的亮度
    image=tf.image.random_brightness(image,0.5)
    #随机改变对比度
    image=tf.image.random_contrast(image,0,1)
    #改变数据类型
    image=tf.cast(image,tf.float32)
    #将图像进行归一化
    image=image/255
    #现在还需要对label进行处理，我们现在是列表[1,2,3],
    #需要变成[[1].[2].[3]]
    label_one_hot = np.zeros(category_num)
    label_one_hot[label] = 1.0
    return image, label_one_hot
    
    
def load_image(image_path, label, category_num):
    #print(image_path)
    #image_path = "/home/ubuntu/shy/tf2_learn/21_pokemon/" + image_path
    #print(image_path)
    image = cv2.imread(image_path.numpy().decode()).astype(np.float32)
    #print(image.shape)
    image = random_size(image, target_size=256)
    image = random_crop(image)
    image = normalize(image)
    label_one_hot = np.zeros(category_num)
    label_one_hot[label] = 1.0

    return image, label_one_hot


def train_iterator(train_list_path, class_nums, batch_size):
    # train_data_path璺緞TXT鏂囦欢
    # images, labels = load_list(train_data_path) # jfkdsjf.jpg 2
    images, labels = load_list(train_list_path)  # jfkdsjf.jpg 2
    print(images)
    #img = []
    #for image in images:
    #  image = "/home/ubuntu/shy/tf2_learn/21_pokemon/" + image
     # img.append(image)
    # print(images)
    print(len(images), len(labels))
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(
        lambda x, y: tf.py_function(load_image, inp=[x, y, class_nums], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.shuffle(len(images))  # 娴嬭瘯闆嗕笉闇�瑕?
    dataset = dataset.repeat()  # 娴嬭瘯闆嗕笉闇�瑕?
    dataset = dataset.batch(batch_size)
    it = dataset.__iter__()
    #return it, images
    return it

def test_iterator(test_list_path, class_nums, batch_size):
    images, labels = load_list(test_list_path)  # jfkdsjf.jpg 2
    #print(images, labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(
        lambda x, y: tf.py_function(load_image, inp=[x, y, class_nums], Tout=[tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    it = dataset.__iter__()
    #return it, images
    #print(images)
    return it
