weight_decay = 1e-4
epoch_num = 50
hash_bit = 512
num_class = 1000
input_shape = (224, 224, 3)
label_smoothing = 0.1

# training config
batch_size = 128
train_num = 1281167 #6149
iterations_per_epoch = int(train_num / batch_size) + 1
warm_iterations = iterations_per_epoch
print(iterations_per_epoch)


initial_learning_rate = 0.05
minimum_learning_rate = 0.0001

# test config
test_batch_size = 128
test_num = 50000
test_iterations = int(test_num / test_batch_size) + 1

log_file = 'result/imagenet/Resnet-512-4.18.txt'

# lr
learning_rate = [0.1, 0.01, 0.001]
boundaries = [80 * iterations_per_epoch, 120 * iterations_per_epoch]


# path
train_data_path = '/root/shy/dataset/imagenet/ILSVRC2012_img_train/IMAGE_NET_Train/'
test_data_path = '/root/shy/dataset/imagenet/ILSVRC2012_img_val/'
imagenet_train_path = '../dataset/imagenet/train_label.txt'
imagenet_test_path = '../dataset/imagenet/validation_label.txt'

flowers17_train_path = "./flowers17/train.txt"
flowers17_test_path = "./flowers17/test.txt"

flowers102_train_path = "./oxford-102-flowers/train.txt"
flowers102_test_path = "./oxford-102-flowers/test.txt"

sun397_train_path = "../dataset/imagenet/train_label.txt"
sun397_test_path = "../dataset/imagenet/validation_label.txt"

short_side_scale = (256, 384)
aspect_ratio_scale = (0.8, 1.25)
hue_delta = (-36, 36)
saturation_scale = (0.6, 1.4)
brightness_scale = (0.6, 1.4)
pca_std = 0.1

mean = [103.939, 116.779, 123.68]
std = [58.393, 57.12, 57.375]
eigval = [55.46, 4.794, 1.148]
eigvec = [[-0.5836, -0.6948, 0.4203],
          [-0.5808, -0.0045, -0.8140],
          [-0.5675, 0.7192, 0.4009]]
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--BATCH_SIZE', type=int, default=512)                                   # 一次训练所取的样本数  128 -> 512
parser.add_argument('--LR', type=float, default=0.01)                                         # 学习率  0.1 -> 0.01
parser.add_argument('--weight_decay', type=float, default=5e-4)                              # 权重衰减，惩罚系数
parser.add_argument('--momentum', type=float, default=0.9)                                   # 动量
parser.add_argument('--epochs', type=int, default=1350)                                       # 训练次数：res:200; dense:300
parser.add_argument('--print_intervals', type=int, default=10)                              # 测试间隔：1024-20,10000-2
parser.add_argument('--data_dir', type=str, default='F:\粒球深度学习\dataset\cifar10')                # 数据集地址
# parser.add_argument('--evaluation', type=bool, default=False)
parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')  # 模型参数保存
# parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--purity', type=float, default=0.80)
parser.add_argument('--recluster', type=int, default=6, help='内部再聚类次数')
args = parser.parse_args()
