
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--BATCH_SIZE', type=int, default=128)                                   # 一次训练所取的样本数  128 -> 512
parser.add_argument('--LR', type=float, default=0.1)                                         # 学习率  0.1 -> 0.01
parser.add_argument('--weight_decay', type=float, default=5e-4)                              # 权重衰减，惩罚系数
parser.add_argument('--momentum', type=float, default=0.9)                                   # 动量
parser.add_argument('--epochs', type=int, default=100)                                       # 训练次数：res:200; dense:300
parser.add_argument('--print_intervals', type=int, default=10)                              # 测试间隔：1024-20,10000-2
parser.add_argument('--data_dir', type=str, default='F:\粒球深度学习\dataset\cifar10')                # 数据集地址
# parser.add_argument('--evaluation', type=bool, default=False)
parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')  # 模型参数保存
# parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--purity', type=float, default=0.80)
parser.add_argument('--recluster', type=int, default=6, help='内部再聚类次数')
args = parser.parse_args()
