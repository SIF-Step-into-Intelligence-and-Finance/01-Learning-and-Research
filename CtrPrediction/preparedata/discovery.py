#查看文件有多少行
# num_line = sum([1 for l in open("../dataset/Electronics/Electronics_test_V1", "r")])
# print(num_line)
# num_line = sum([1 for l in open("../dataset/Electronics/Electronics_valid_V1", "r")])
# print(num_line)
# file=open("../dataset/taobao/taobao_last_samples", "r")
# user={}
# count=0
# for line in file:
#     items=line.strip().split(',')
#     if(items[2] not in user):
#        user[items[2]]=0
#        count+=1
# print(count)


# 查看大于某个时间段的评论数
# file = open('../dataset/Clothing/Clothing_sorted_with_neg')
# time = {}
# a=0
# counts=0
# for line in file:
#     items = line.strip().split(',')
#     if items[0] == '1':
#         counts+=1
#         if items[4]>='1397116033':
#             a += 1
# print(a)
# print(counts)
# file = open('../dataset/Electronics/Electronics_last_samples')
# time = {}
# a=0
# counts=0
# for line in file:
#     items = line.strip().split(',')
#     if items[1] == '1':
#         counts += 1
#         if items[5]>='1397116033':
#             a += 1
# print(a)
# print(counts)
# file = open('../dataset/Electronics/Electronics_last_samples')
# time = {}
# a=0
# counts=0
# for line in file:
#     items = line.strip().split(',')
#     if items[1] == '1':
#         counts+=1
#         if items[5]>='1397116033':
#             a += 1
# print(counts)
# print(a)

#查看训练集和测试集中用户的数目
# import os
# epsilon = 0.000000001
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# category = '../dataset/Clothing/'
# def datafilename(datapath, file_name):
#   return  datapath  + file_name
#
# fi = open(datafilename(category, "Clothing_last_samples"), "r")
# ftrain = open(datafilename(category, "Clothing_train_V2"), "r")
# ftest = open(datafilename(category, "Clothing_valid_V2_part"), "r")
# user_map={}
# count=0
# for line in fi:
#     items = line.strip().split(',')
#     if items[2] not in user_map:
#        user_map[items[2]]=0
#        count+=1
# print(count)
# user_map_train={}
# count=0
# for line in ftrain:
#     items = line.strip().split(',')
#     if items[1] not in user_map_train:
#        user_map_train[items[1]]=0
#        count+=1
# print(count)
# user_map_test={}
# count=0
# for line in ftest:
#     items = line.strip().split(',')
#     if items[1] not in user_map_test:
#        user_map_test[items[1]]=0
#        count+=1
# print(count)
# count1=0
# for (key,value) in user_map_test.items():
#     if key not in user_map_train.keys():
#         count1+=1
# print(count1)
