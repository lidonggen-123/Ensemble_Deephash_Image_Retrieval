import numpy as np
from tqdm import tqdm
import time

np.set_printoptions(threshold=np.inf)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def hanming(v1, v2):
    tt = 1.0 * np.sum(v1 != v2, axis=1)
    return tt


def true_divide(buffer_yes, Ns):
    P = np.true_divide(np.cumsum(buffer_yes), np.array(Ns))
    return P


def argsort(similarity):
    return np.argsort(similarity, kind='mergesort')


def sum(arr):
    res = 0
    for i in range(len(arr)):
        res += arr[i]
    return res


def hanming_normalazition(arr):
    res = []
    # print("汉明归一化循环次数，",len(arr))
    for i in range(len(arr)):
        res.append(round(1 - (arr[i] / 48), 4))
    return res
    # return [round(1 - (i / 48), 4) for i in arr]


def cumsum(arr):
    return np.cumsum(arr)


def zeros(K):
    return np.zeros(K)


def fun(label, y, K, query_label, buffer_yes):
    # print("fun函数  循环次数：", K)
    for j in range(0, K):
        # print(j)
        # 检索出来的标签
        retrieval_label = label[y[j]]
        if query_label == retrieval_label:
            buffer_yes[j] = 1
    return buffer_yes


def produce_hanming_list(arr):
    temp = []
    for i in range(len(arr)):
        temp.append(arr[i])
    return temp


def hanming_matrix(hanming_normalazition):
    hanming = []
    for i in hanming_normalazition:
        hanming.append(hanming_normalazition[i])
    return hanming


def list_add(K, pic_path1, pic_path2, pic_path3, hanming_matrix1, hanming_matrix2, hanming_matrix3):
    list1_1, list1_2, list1_3, = [], [], []
    # 三个for循环比较慢
    # 键 1-n   值:正确图片的索引
    # print("pic_path",pic_path1)
    # print(pic_path1.values())
    for k, v in pic_path1.items():
        # print("pic_path1.index(p1)",pic_path1.get(v))
        # print("v",v)
        # print(pic_path1[k])
        result1 = (hanming_matrix1[pic_path1[k]] + hanming_matrix2[pic_path2[k]] + hanming_matrix3[
            pic_path3[k]]) / 3
        list1_1.append(result1)
    # print(list1_1)
    for k, v in pic_path2.items():
        result2 = (hanming_matrix1[pic_path1[k]] + hanming_matrix2[pic_path2[k]] + hanming_matrix3[
            pic_path3[k]]) / 3
        list1_2.append(result2)
    for k, v in pic_path3.items():
        result3 = (hanming_matrix1[pic_path1[k]] + hanming_matrix2[pic_path2[k]] + hanming_matrix3[
            pic_path3[k]]) / 3
        list1_3.append(result3)
    path = list(pic_path1.keys())[:K] + list(pic_path2.keys())[:K] + list(pic_path3.keys())[:K]
    new_list = list1_1 + list1_2 + list1_3
    return path, new_list


def fn(train_label, train_base_dic, final_path_dic, final_index_dict):
    # print("final_path", final_path)
    # base_x = train_base[0].tolist()
    # base_x = [int(i) for i in base_x]   # base_x   0-49999
    temp_x = []
    for _, v in final_index_dict.items():
        # index_x = final_path_dic.get(v)
        temp_x.append(train_label[train_base_dic.get(final_path_dic.get(v))])

    return temp_x[::-1]


def precision(train_label, train_binary, train_base, test_label, test_binary, test_query, top_k):
    """
    train_label:训练集图片的标签
    train_binary:训练集图片的二值码
    test_label:测试集图片的标签
    test_binary:测试集图片的二值码
    top_k:查询返回图片（top_k）数量的map值
    """
    K = top_k
    QueryTimes = test_binary[0].shape[0]  # 获得行数
    AP_1 = np.zeros((QueryTimes, 1))
    AP_2 = np.zeros((QueryTimes, 1))
    AP_3 = np.zeros((QueryTimes, 1))

    # [1,2,....K]
    Ns = [i for i in range(1, K + 1)]
    train_base = train_base[0].tolist()
    # print("train_base",train_base)
    train_base = [int(i) for i in train_base]
    train_base_dic = {}
    for i in range(len(train_base)):
        train_base_dic[i] = i
    # print("train_base",train_base_dic)
    ap1 = 0
    for i in tqdm(range(0, QueryTimes)):
        # for i in range(0, QueryTimes):

        # 查询集   真实标签
        query_label_3 = test_label[2][i]

        print("query_label_3", query_label_3)
        # print(test_query[i])

        # 真实标签的哈希码
        query_binary_1 = test_binary[0][i]  # 去一个测试二值码
        query_binary_2 = test_binary[1][i]
        query_binary_3 = test_binary[2][i]

        # (6144,)  汉明矩阵  乱序

        similarity_1 = hanming(train_binary[0], query_binary_1)

        similarity_2 = hanming(train_binary[1], query_binary_2)
        similarity_3 = hanming(train_binary[2], query_binary_3)

        # 汉明矩阵排序，返回汉明距离从小到大的索引  即正确的索引

        y2_1 = argsort(similarity_1)

        y2_2 = argsort(similarity_2)
        y2_3 = argsort(similarity_3)
        # (6144,)
        # print(y2_1[:10])

        buffer_yes_1 = zeros(K)

        buffer_yes_2 = zeros(K)
        buffer_yes_3 = zeros(K)

        buffer_yes_1 = fun(train_label[0], y2_1, K, test_label[0][i], buffer_yes_1)
        buffer_yes_2 = fun(train_label[1], y2_2, K, test_label[1][i], buffer_yes_2)
        buffer_yes_3 = fun(train_label[2], y2_3, K, test_label[2][i], buffer_yes_3)

        P_1 = true_divide(buffer_yes_1, Ns)
        P_2 = true_divide(buffer_yes_2, Ns)
        P_3 = true_divide(buffer_yes_3, Ns)

        if sum(buffer_yes_1) == 0:
            AP_1[i] = 0
        else:
            AP_1[i] = sum(np.array(P_1) * np.array(buffer_yes_1)) / sum(buffer_yes_1)

        if sum(buffer_yes_2) == 0:
            AP_2[i] = 0
        else:
            AP_2[i] = sum(np.array(P_2) * np.array(buffer_yes_2)) / sum(buffer_yes_2)

        if sum(buffer_yes_3) == 0:
            AP_3[i] = 0
        else:
            AP_3[i] = sum(np.array(P_3) * np.array(buffer_yes_3)) / sum(buffer_yes_3)

        # 排序后的汉明距离矩阵

        order_map_1 = [similarity_1[i] for i in y2_1]

        # print("order_map_1", order_map_1)
        order_map_2 = [similarity_2[i] for i in y2_2]
        order_map_3 = [similarity_3[i] for i in y2_3]

        pic_path1 = {}
        pic_path2 = {}
        pic_path3 = {}

        for k, v in enumerate(y2_1, 0):
            # print("k",k)
            # print("v",v)
            pic_path1[v] = k

        # for step,k in enumerate(y2_1):
        #     print("step",step)
        #     print("k",k)
        #     pic_path1[k] = step
        for k, v in enumerate(y2_2, 0):
            pic_path2[v] = k
        for k, v in enumerate(y2_3, 0):
            pic_path3[v] = k

        hanming_normalazition1 = hanming_normalazition(order_map_1)

        hanming_normalazition2 = hanming_normalazition(order_map_2)
        hanming_normalazition3 = hanming_normalazition(order_map_3)

        # print("hanming_normalazition1", hanming_normalazition1)

        # 汉明距离归一化 * 精度
        hanming_matrix1 = [i * acc[0] for i in hanming_normalazition1]
        hanming_matrix2 = [i * acc[1] for i in hanming_normalazition2]
        hanming_matrix3 = [i * acc[2] for i in hanming_normalazition3]

        path, new_list = list_add(K, pic_path1, pic_path2, pic_path3, hanming_matrix1, hanming_matrix2, hanming_matrix3)

        t1 = time.time()

        dictionary = {}
        for step, p in enumerate(path):
            value = dictionary.get(p, [])
            value.append(step)
            dictionary[p] = value
        # print("dictionary",dictionary)
        path_arr = [None for _ in range(len(path))]
        num_arr = [0 for _ in range(len(path))]

        for k, v in dictionary.items():
            path_arr[v[0]] = k
            num_arr[v[0]] = new_list[v[0]]

        final_path = [i for i in path_arr if i is not None][:K]
        final_path_dic = {}
        for step, r in enumerate(final_path):
            final_path_dic[step] = r
        # print("final_path_dic",final_path_dic)
        final_num = [i for i in num_arr if i][:K]

        final_index = argsort(final_num)
        final_index_dict = {}
        for step, r in enumerate(final_index):
            final_index_dict[step] = r

        lable_final = fn(train_label[0], train_base_dic, final_path_dic, final_index_dict)

        # print("lable_final", lable_final)
        count = 0
        sum_value = 0
        for step, r in enumerate(lable_final[:top_k]):
            step += 1
            if str(query_label_3) == r:
                count += 1
                sum_value += count / step
        if count:
            ap = sum_value / count
        else:
            ap = 0
        ap1 += ap
        print("ap", ap)
        print("AP_1[i]", AP_1[i])
        print("AP_2[i]", AP_2[i])
        print("AP_3[i]", AP_3[i])
        t2 = time.time()
        print("AP时间", t2 - t1)

    map_1 = np.mean(AP_1)
    map_2 = np.mean(AP_2)
    map_3 = np.mean(AP_3)
    ap1 = ap1 / QueryTimes

    # return precision_at_k, map
    return map_1, map_2, map_3, ap1


if __name__ == '__main__':
    # vgg resnet dense
    acc = [0.669, 0.692, 0.705]
    # 测试集
    # F:\project\produce_hash\NPY
    # F:\project\produce_hash\MAP_集成.py
    TestData_VGG = np.load("NPY/sun397/48bit/VGG_npy/CheckData/Code_Check.npy", allow_pickle=True)
    TestLabel_VGG = np.load("NPY/sun397/48bit/VGG_npy/CheckData/Code_label.npy", allow_pickle=True)
    TestPath_VGG = np.load("NPY/sun397/48bit/VGG_npy/CheckData/Code_Path.npy", allow_pickle=True)
    # 训练集
    DataBase_VGG = np.load("NPY/sun397/48bit/VGG_npy/DataBase/Code_Base.npy", allow_pickle=True)
    DataLabel_VGG = np.load("NPY/sun397/48bit/VGG_npy/DataBase/Code_number.npy", allow_pickle=True)
    DataPath_VGG = np.load("NPY/sun397/48bit/VGG_npy/DataBase/Code_Pic.npy", allow_pickle=True)

    # 测试集
    TestData_Resnet = np.load("NPY/sun397/48bit/Resnet_npy/CheckData/Code_Check.npy", allow_pickle=True)
    TestLabel_Resnet = np.load("NPY/sun397/48bit/Resnet_npy/CheckData/Code_label.npy", allow_pickle=True)
    TestPath_Resnet = np.load("NPY/sun397/48bit/Resnet_npy/CheckData/Code_Path.npy", allow_pickle=True)
    # 训练集
    DataBase_Resnet = np.load("NPY/sun397/48bit/Resnet_npy/DataBase/Code_Base.npy", allow_pickle=True)
    DataLabel_Resnet = np.load("NPY/sun397/48bit/Resnet_npy/DataBase/Code_number.npy", allow_pickle=True)
    DataPath_Resnet = np.load("NPY/sun397/48bit/Resnet_npy/DataBase/Code_Pic.npy", allow_pickle=True)

    # 测试集
    TestData_Desnet = np.load("NPY/sun397/48bit/Densenet_npy/CheckData/Code_Check.npy", allow_pickle=True)
    TestLabel_Desnet = np.load("NPY/sun397/48bit/Densenet_npy/CheckData/Code_label.npy", allow_pickle=True)
    TestPath_Desnet = np.load("NPY/sun397/48bit/Densenet_npy/CheckData/Code_Path.npy", allow_pickle=True)
    # 训练集
    DataBase_Desnet = np.load("NPY/sun397/48bit/Densenet_npy/DataBase/Code_Base.npy", allow_pickle=True)
    DataLabel_Desnet = np.load("NPY/sun397/48bit/Densenet_npy/DataBase/Code_number.npy", allow_pickle=True)
    DataPath_Desnet = np.load("NPY/sun397/48bit/Densenet_npy/DataBase/Code_Pic.npy", allow_pickle=True)

    DataLabel = [DataLabel_VGG, DataLabel_Resnet, DataLabel_Desnet]
    DataBase = [DataBase_VGG, DataBase_Resnet, DataBase_Desnet]
    DataPath = [DataPath_VGG, DataPath_Resnet, DataPath_Desnet]

    TestLabel = [TestLabel_VGG, TestLabel_Resnet, TestLabel_Desnet]
    TestData = [TestData_VGG, TestData_Resnet, TestData_Desnet]
    TestPath = [TestPath_VGG, TestPath_Resnet, TestPath_Desnet]

    map = precision(DataLabel, DataBase, DataPath, TestLabel, TestData, TestPath, 1000754)

    filewrite = open("map_sun397_48bit_top1000754_bit.txt", "w")
    for i in map:
        filewrite.write("map: " + str(i) + "\n")
        filewrite.write("bit_num: " + "48\n")
        print(i)
