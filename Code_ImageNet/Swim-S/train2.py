# coding=gbk
import torch
from model2 import swin_s
import warnings
import torch.nn as nn
from read_data2 import read_imagenet
import time

warnings.filterwarnings("ignore")
import torch
import numpy as np

weight_decay = 1e-4
from config import args
import torch.nn as nn
import os

# training config
batch_size = 32

iterations_per_epoch = 10011
test_batch_size = 391

epoch_num = 100

# test config
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
cpu_device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = swin_s().to(device)
# 分别提取模型的权重参数和偏置参数


optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=args.momentum, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=args.momentum, weight_decay=args.weight_decay)
# Q3 学习率变化策略
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 122, 163], gamma=0.1, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24, 48, 68,80], gamma=0.1,
                                                 last_epoch=-1)


def save_checkpoint(best_acc, model, optimizer, epoch):
    print('Best Model Saving...')
    model_state_dict = model.state_dict()
    if not os.path.isdir('checkpoints_712_1024'):
        os.mkdir('checkpoints_712_1024')
    torch.save({
        'model_state_dict': model_state_dict,  # 网络参数
    }, os.path.join('checkpoints_712_1024', 'checkpoint_model_best_{}.pth'.format(best_acc)))

    """
    相关参数
    """


def hash_loss_fn(hash_input):
    loss1 = -1 * torch.mean(torch.square(hash_input - 0.5)) + 0.25
    loss2 = torch.mean(torch.square(torch.mean(hash_input, axis=1) - 0.5))
    hash_loss = loss1 + loss2

    return hash_loss


def L2Loss(y, yhead):
    return torch.mean((y.to(device) - yhead.to(device)) ** 2)
    


def accuracy(y_true, y_pred):
    correct_num = 0
    for k in range(y_true.size()[0]):
        i = torch.argmax(y_true, -1)[k].to(device)
        j = torch.argmax(y_pred, -1)[k].to(device)
        if torch.equal(i, j) == True:
            correct_num += 1

    return correct_num


def train(x, y, i):
      prediction, hashcode = net(x)
      acc = accuracy(y.to(device), prediction.to(device)) / 128
      hash_loss = hash_loss_fn(hashcode)
      ce = cost(prediction.to(device), y.to(device))
      l2 = L2Loss(prediction, y).to(device)
      loss = ce + l2 + hash_loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      return hashcode, loss, ce, l2, hash_loss, acc


def test(x, y):
    with torch.no_grad():
        prediction, hashcode = net(x)
        acc = accuracy(y.to(device), prediction.to(device))
        print(acc / 128)
        return acc / 128


def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()


if __name__ == '__main__':
    since = time.time()  # 用于计算运行时间
    best_acc = 0
    best_acc_loc = 0
    train_dataset_loader, test_dataset_loader= read_imagenet(128)
    print(len(train_dataset_loader))
    for j in range(1, args.epochs + 1):
        net.train()  # 训练
        filewriter = open('hashcode/train_1024_712.txt', 'w')
        filewriter_res = open('result/train_1024_712.txt', 'w')
        sum_accuracy = 0
        count = 0
        cost = nn.CrossEntropyLoss().to(device)
        sum_loss = 0
        sum_ce_loss = 0
        sum_l2_loss = 0
        sum_hash_loss = 0
        count1 = 0  
        for i, (train_img, train_label) in enumerate(train_dataset_loader):
            count1 += 1
            label = torch.zeros(train_label.size()[0], 1000)
            for n, l1 in enumerate(train_label):
                label[n, l1.numpy()] = float(1)
            train_label = label
            hashcode, cur_loss, ce, l2, hash_loss, acc = train(train_img, train_label, i)
            sum_loss += cur_loss
            sum_ce_loss += ce
            sum_l2_loss += l2
            sum_hash_loss += hash_loss
            if i % 100== 0 and i > 1:
                print("周期：%d，loss:%f,ce:%f,l2:%f,hash:%f，acc%f" % (
                    j, sum_loss / (count1), sum_ce_loss / (count1), sum_l2_loss / (count1), sum_hash_loss / (count1)
                    , acc))
            if i % 1000 == 0 and i > 1:
                cur_time = time.time()
                it_time = cur_time - since

                cou = (i / iterations_per_epoch + j) / args.epochs
                total_time = it_time / cou
                rest_time = total_time - it_time
                print("当前用时：%d h,%d min,%d s,剩余用时：%d h,%d min,%d s。" % (int(it_time / 3600),
                                                                        int((it_time % 3600) / 60),
                                                                        int(((it_time % 3600) % 60) % 60),
                                                                        int(rest_time / 3600),
                                                                        int((rest_time % 3600) / 60),
                                                                        int(((rest_time % 3600) % 60) % 60)
                                                                        )

                      )
            sumacc = 0
            if i == 10000:
                print("test")
                net.eval()  # 进入测试，测试时不启用 Batch Normalization 和 Dropout
                num_correct = 0.
                test_total = 0
                for k, (test_images, test_labels) in enumerate(test_dataset_loader):
                    label = torch.zeros(test_labels.size()[0], 1000)
                    for n, l1 in enumerate(test_labels):
                        label[n, l1.numpy()] = float(1)
                    test_labels = label
                    acc1 = test(test_images, test_labels)
                    sumacc += acc1
                sumacc = sumacc / k
                print("当前精度%8f" % (sumacc))
                if sumacc > best_acc:
                    best_acc = sumacc
                    best_acc_loc = j
                    sum_loss = 0
                    sum_ce_loss = 0
                    sum_l2_loss = 0
                    sum_hash_loss = 0
                    count1 = 0
                    if j == 1 or j==2:
                        save_checkpoint(best_acc, net, optimizer, j)
                    if best_acc >= 0.20 :
                        save_checkpoint(best_acc, net, optimizer, j)
                net.train()  # 返回训练
            if i % 1000 == 0:
                print("周期，batch数：当前acc,最好acc,最好acc周期", j, i, sumacc, best_acc, best_acc_loc)
                print('Current Learning Rate: {}'.format(scheduler.get_last_lr()))          
        scheduler.step()
          
