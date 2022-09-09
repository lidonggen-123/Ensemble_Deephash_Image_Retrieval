import numpy as np

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
 
    for i in range(len(arr)):
        res.append(round(1 - (arr[i] / 1024), 4))
    return res
    



def cumsum(arr):
    return np.cumsum(arr)



def zeros(K):
    return np.zeros(K)


def fun(label, y, K, query_label, buffer_yes):
    
    for j in range(0, K):
        # print(j)
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
    count=0
    for k, v in pic_path1.items():
        # print("pic_path1.index(p1)",pic_path1.get(v))
        # print("v",v)
        # print(pic_path1[k])
        result1 = hanming_matrix1[pic_path1[k]] 
        list1_1.append(result1)
        count+=1
        if count==1000:
          count=0
          break
    print(np.mean(list1_1))
    # print(list1_1)
    for k, v in pic_path2.items():
        result2 =  hanming_matrix2[pic_path2[k]] 
        list1_2.append(result2)
        count+=1
        if count==1000:
          count=0
          break
    print(np.mean(list1_2))
    for k, v in pic_path3.items():
        result3 = hanming_matrix3[pic_path3[k]]
        list1_3.append(result3)
        count+=1
        if count==1000:
          count=0
          break
    print(np.mean(list1_3))
    ave=(np.mean(list1_1)+np.mean(list1_2)+np.mean(list1_3))/3
    a1=np.linspace(np.mean(list1_1)-ave,np.mean(list1_1)-ave,1000)
    a2=np.linspace(np.mean(list1_2)-ave,np.mean(list1_2)-ave,1000)
    a3=np.linspace(np.mean(list1_3)-ave,np.mean(list1_3)-ave,1000)
    b1=np.linspace(acc[0],acc[0],1000)
    b2=np.linspace(acc[1],acc[1],1000)
    b3=np.linspace(acc[2],acc[2],1000)
    
    list1_1=(list1_1-a1)*b1
    list1_2=(list1_2-a2)*b2
    list1_3=(list1_1-a3)*b3
    new_list=[]
    print(list1_1)
    print("dddddddddddddddddddddddddd")
    print(list1_2)
    print("dddddddddddddddddddddddddd")
    print(list1_3)
    print("dddddddddddddddddddddddddd")
    path = list(pic_path1.keys())[:K] + list(pic_path2.keys())[:K] + list(pic_path3.keys())[:K]
    new_list=np.concatenate([list1_1,list1_2,list1_3])
    return path, new_list


def fn(train_label, train_label1,train_base_dic, final_path_dic, final_index_dict):
    # print("final_path", final_path)
    # base_x = train_base[0].tolist()
    # base_x = [int(i) for i in base_x]   # base_x   0-49999
    temp_x = []
    temp_x1 = []
    for _, v in final_index_dict.items():
        # index_x = final_path_dic.get(v)
        temp_x.append(train_label[train_base_dic.get(final_path_dic.get(v))])
        temp_x1.append(train_label1[train_base_dic.get(final_path_dic.get(v))])
    return temp_x[::-1],temp_x1[::-1]



def precision(train_label, train_binary, train_base, test_label, test_binary, test_query, top_k):
    print("start")
    K = top_k
    QueryTimes = 100
    AP_1 = np.zeros((100, 1))
    AP_2 = np.zeros((100, 1))
    AP_3 = np.zeros((100, 1))

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
    li=[  44,  97,  106, 126,  141,  194, 210, 269,  312, 329, 356, 360,369, 409,410,489,523,  638,  642,
  671,  711,    798,  801,   838,  860,960, 1069, 1160,1190, 1491
 , 1616, 1665, 2053, 2187, 2571, 2722, 2779, 2820, 2839, 2917, 2972,
 3012, 3195, 3309, 3501, 3542, 3571, 3613, 3639, 3703, 3738, 3781,
 3806, 3835, 3852, 3858, 3868, 3917, 3951, 3961, 3978, 3996, 4009, 4097,
 4118, 4124, 4204, 4230, 4257, 4293, 4297,  4371, 4392, 4430, 4524,
 4545, 4711, 4796, 4852, 4934, 4966, 5020, 5330, 5394, 5503, 5862, 5912,
 6008, 6402, 6492, 6648, 7320, 7431,  7492, 7654, 7832, 
 8318, 8383, 8390, 8675]
    count1=0
    for i in li:
        print(i)
        # for i in range(0, QueryTimes):
        query_label_1 = test_label[0][i]
        query_label_3 = test_label[2][i]

        print("query_label_3", query_label_3)
        # print(test_query[i])

      
        query_binary_1 = test_binary[0][i] 
        query_binary_2 = test_binary[1][i]
        query_binary_3 = test_binary[2][i]

        

        similarity_1 = hanming(train_binary[0], query_binary_1)

        similarity_2 = hanming(train_binary[1], query_binary_2)
        similarity_3 = hanming(train_binary[2], query_binary_3)

        

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
            AP_1[count1] = 0
        else:
            AP_1[count1] = sum(np.array(P_1) * np.array(buffer_yes_1)) / sum(buffer_yes_1)

        if sum(buffer_yes_2) == 0:
            AP_2[count1] = 0
        else:
            AP_2[count1] = sum(np.array(P_2) * np.array(buffer_yes_2)) / sum(buffer_yes_2)

        if sum(buffer_yes_3) == 0:
            AP_3[count1] = 0
        else:
            AP_3[count1] = sum(np.array(P_3) * np.array(buffer_yes_3)) / sum(buffer_yes_3)

       

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

     
        hanming_matrix1 = [i  for i in hanming_normalazition1]
        hanming_matrix2 = [i  for i in hanming_normalazition2]
        hanming_matrix3 = [i  for i in hanming_normalazition3]
        path, new_list = list_add(K, pic_path1, pic_path2, pic_path3, hanming_matrix1, hanming_matrix2, hanming_matrix3)
    
    
        t1 = time.time()
        # pat15000
        dictionary = {}
        # 去重
        for step, p in enumerate(path):
            value = dictionary.get(p, [])
            value.append(step)
            dictionary[p] = value
        path_arr = [None for _ in range(len(path))]
        num_arr = [0 for _ in range(len(path))]
        for k, v in dictionary.items():
            path_arr[v[0]] = k
            num_arr[v[0]] = new_list[v[0]]
        final_path = [i for i in path_arr if i is not None][:len(path)]
        final_path_dic = {}
        for step, r in enumerate(final_path):
            final_path_dic[step] = r
        final_num = [i for i in num_arr if i][:len(path)]
        # 排名后的300，从小到大
        final_index = argsort(final_num)
        final_index_dict = {}
        for step, r in enumerate(final_index):
            final_index_dict[step] = r
        lable_final, lable_final1 = fn(train_label[0], train_label[1], train_base_dic, final_path_dic,
                                                    final_index_dict)
        
 
        # print("lable_final", lable_final)
        count = 0
        sum_value = 0
        for step, r in enumerate(lable_final[:top_k]):
            r2 = lable_final1[step]
            step += 1
            print(query_label_3,query_label_1,r,r2)
            if ((str(query_label_3) == r) or (str(query_label_1) == r)) or (
                    (str(query_label_1) == r2) or (str(query_label_3) == r2)):
                count += 1
                sum_value += count / step
        if count:
            ap = sum_value / count
        else:
            ap = 0
        ap1 += ap
        print("ap", ap)
        print("AP_1[i]", AP_1[count1])
        print("AP_2[i]", AP_2[count1])
        print("AP_3[i]", AP_3[count1])
        t2 = time.time()
        print(np.mean(AP_1))
        print(np.mean(AP_2))
        print(np.mean(AP_3))
        print(ap1 / 100)
        count1+=1
    map_1 = np.mean(AP_1)
    map_2 = np.mean(AP_2)
    map_3 = np.mean(AP_3)
    ap1 = ap1 / 100

    # return precision_at_k, map
    return map_1, map_2, map_3, ap1


if __name__ == '__main__':
    # vgg resnet dense
    acc = [0.774, 0.735, 0.753]
    print("111")
    TestData_VGG = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Swim_transformer_npy/CheckData/Code_Check.npy", allow_pickle=True)
    TestLabel_VGG = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Swim_transformer_npy/CheckData/Code_label.npy", allow_pickle=True)
    TestPath_VGG = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Swim_transformer_npy/CheckData/Code_Path.npy", allow_pickle=True)
    print("222")
    DataBase_VGG = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Swim_transformer_npy/DataBase/Code_Base.npy", allow_pickle=True)
    print("dd")
    DataLabel_VGG = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Swim_transformer_npy/DataBase/Code_number.npy", allow_pickle=True)
    print("dd")
    DataPath_VGG = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Swim_transformer_npy/DataBase/Code_Pic.npy", allow_pickle=True)
    print("333")
    TestData_Resnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/VGG_npy/CheckData/Code_Check.npy", allow_pickle=True)
    TestLabel_Resnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/VGG_npy/CheckData/Code_label.npy", allow_pickle=True)
    TestPath_Resnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/VGG_npy/CheckData/Code_Path.npy", allow_pickle=True)
    print("444")
    DataBase_Resnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/VGG_npy/DataBase/Code_Base.npy", allow_pickle=True)
    DataLabel_Resnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/VGG_npy/DataBase/Code_number.npy", allow_pickle=True)
    DataPath_Resnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/VGG_npy/DataBase/Code_Pic.npy", allow_pickle=True)
    print("555")
    TestData_Desnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Densenet_npy/CheckData/Code_Check.npy", allow_pickle=True)
    TestLabel_Desnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Densenet_npy/CheckData/Code_label.npy", allow_pickle=True)
    TestPath_Desnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Densenet_npy/CheckData/Code_Path.npy", allow_pickle=True)
    print("666")
    DataBase_Desnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Densenet_npy/DataBase/Code_Base.npy", allow_pickle=True)
    DataLabel_Desnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Densenet_npy/DataBase/Code_number.npy", allow_pickle=True)
    DataPath_Desnet = np.load("/home/ubuntu/lxd-workplace/shy/hashcode_imagenet_1024bit/NPY/imagenet/1024bit/Densenet_npy/DataBase/Code_Pic.npy", allow_pickle=True)
    print("777")
    DataLabel = [DataLabel_VGG, DataLabel_Resnet, DataLabel_Desnet]
    DataBase = [DataBase_VGG, DataBase_Resnet, DataBase_Desnet]
    DataPath = [DataPath_VGG, DataPath_Resnet, DataPath_Desnet]

    TestLabel = [TestLabel_VGG, TestLabel_Resnet, TestLabel_Desnet]
    TestData = [TestData_VGG, TestData_Resnet, TestData_Desnet]
    TestPath = [TestPath_VGG, TestPath_Resnet, TestPath_Desnet]
    map = precision(DataLabel, DataBase, DataPath, TestLabel, TestData, TestPath, 1000)

    filewrite = open("map_IMAGNET_32bit_top50000_bit_628.txt", "w")
    for i in map:
        filewrite.write("map: " + str(i) + "\n")
        filewrite.write("bit_num: " + "32\n")
        print(i)

