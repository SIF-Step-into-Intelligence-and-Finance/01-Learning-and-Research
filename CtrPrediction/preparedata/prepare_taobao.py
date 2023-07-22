# -*- coding: UTF-8 -*-
'''
user_id,item_id,item_category,behavior_type,time
'''

import random
import numpy as np
from tqdm import tqdm
import pickle

np.random.seed(1234)
random.seed(1234)
category = '../dataset/taobao/'
#淘宝只有一个reviews-info文件# 100,2268318,2520377,pv,1511544070
# -----------------------------------------taobao-item-info-----------------------------------------
def create_iteminfo():
    file_review = open("../dataset/taobao/reviews-info", "r")
    file_iteminfo = open("../dataset/taobao/item-info", "w")
    items_map = set()
    for line in file_review:
        items = line.strip().split(",")  # [1,2268318,2520377,1511544070]
        if items[1] not in items_map:
            items_map.add(items[1])
            print(items[1] + ',' + items[2], file=file_iteminfo)
# -----------------------------------------taobao_sorted_with_neg-----------------------------------------
# 这里已经按用户排好序了，也就是说文件中用户行为是连着的，且行为按时间排序
def manual_join():
    with open(f"../dataset/taobao/reviews-info", "r") as file_review:
        user_map = {}
        item_list = []
        num_lines = sum(1 for _ in open("../dataset/taobao/reviews-info", "r"))
        for line in tqdm(file_review, total=num_lines, ncols=80):
            items = line.strip().split(",")  # [reviewerID,asin,category,unixReviewTime]
            if items[0] not in user_map:
                user_map[items[0]] = []
            user_map[items[0]].append((items, float(items[-1])))
            item_list.append(items[1])

    # 构造物品行为序列，曾经购买过该物品的用户 :[用户，时间]{'0000031887': [('A1KLRMWW2FWPL4', 1297468800.0), ('A2G5TCU2WDFZ65', 1358553600.0)]}
    file_review = open(f"../dataset/taobao/reviews-info", "r")
    useridToClickItem = {}
    for line in file_review:
        items = line.strip().split(",")  # A1KLRMWW2FWPL4	0000031887	5.0	1297468800
        if items[1] not in useridToClickItem:
            useridToClickItem[items[1]] = []
        useridToClickItem[items[1]].append((items[0], float(items[-1])))  # data[0]是用户，data[-1]是时间
    #对物品的行为序列进行排序
    useridToClickItem = {key: sorted(value, key=lambda x: x[1]) for key, value in useridToClickItem.items()}

    # item_map：（itemID，categoryID）
    file_meta = open(f"../dataset/taobao/item-info", "r")
    item_map = {}
    for line in file_meta:
        arr = line.strip().split(",")
        if arr[0] not in item_map:
            item_map[arr[0]] = arr[1]

    file_output = open(f"../dataset/taobao/taobao_sorted_with_neg", "w")
    # gap = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    for key in tqdm(user_map):  # 遍历所有用户
        #这里不用排序，因为已经拍好序了
        # sorted_user_bh = sorted(user_map[key], key=lambda x: x[1])  # 先对该用户的行为按时间从前到后排序，这里排序只是为了使文件每一行有序，并没有组合成一行的行为序列[('A2RN8W7U1IHSKC,B004YTJF9U,1366156800', 1366156800.0)...,...]
        for items, _ in user_map[key]:  # 对该用户的每个行为构造负样本  # A1KLRMWW2FWPL4	0000031887	1297468800
            asin = items[1]
            negative_num = 0
            # 负样本的构造
            while True:
                asin_neg_index = random.randint(0, len(item_list) - 1)
                asin_neg = item_list[asin_neg_index]
                if asin_neg == asin:
                    continue
                # 去判断一下有没有在当前时间之前购买的用户，算作物品的历史行为(也有可能该用户是第一个购买的，所以物品的历史行为仍然为0，下面还要处理)
                user_list = []
                user_str = ""
                user_time_list = []
                user_time_str = ""
                for (user, time) in useridToClickItem[asin_neg]:
                    if (int(time) > int(items[-1])):  # 如果找到一个用户购买该物品的时间在该用户购买该物品的时间之后，那么后边的都不是物品行为序列了（因为时间超了）
                        break  # 直接break的原因就是已经排好序，后边的物品肯定也满足条件
                    if user == items[0]:  # 这个用户购买的不算，也就是说该用户有可能已经购买过该物品？但是这里不算之前购买过的？
                        continue
                    user_list.append(user)
                    user_time_list.append(str(time))
                    # user_tiv = float(days) - time // 3600 // 24 + 1.  # 当前点击时间与之前点击时间相差的天数
                    # user_tiv_str += str(np.sum(user_tiv >= gap)) + ""  # [ True  True  True  True  True  True False False False False False False]->6

                if len(user_str) > 0:
                    user_str ="".join(user_list[-5:])
                    user_time_str = "".join(user_time_list[-5:])
                if len(user_str) == 0:  # 如果该物品在此用户购买之前没有用户购买过，也就是该用户是第一个购买该物品的用户，那么赋予一个默认的用户，保证文件中此项不为空
                    user_str = "default_user"
                    user_time_str = "-1"

                if asin_neg in item_map:
                    print("0" + "," + items[0] + "," + asin_neg + "," + item_map[asin_neg] + "," + items[2] + "," + user_str + "," + user_time_str , file=file_output)
                else:
                    print("0" + "," + items[0] + "," + asin_neg + "," + "default_cat" + "," + items[2] + "," + user_str + "," + user_time_str , file=file_output)
                negative_num += 1
                if negative_num == 1:  #负样本的数目
                    break
            # 正样本
            user_list = []
            user_str = ""
            user_time_list = []
            user_time_str = ""

            for (user, time) in useridToClickItem[asin]:
                if int(time) > int(items[-1]):
                    break
                if user == items[0]:
                    continue
                user_list.append(user)
                user_time_list.append(str(time))

            if len(user_list) > 0:
                user_str = "".join(user_list[-5:])
                user_time_str = "".join(user_time_list[-5:])
            if len(user_str) == 0:
                user_str = "default_user"
                user_time_str = "-1"
            if asin in item_map:#用户id，目标物品id，目标物品类别id，购买目标物品时间，在此时间之前购买过该物品的用户，在此时间之前购买过该物品的用户的购买时间
                print("1" + "," + items[0] + "," + items[1] + "," + items[2]+ "," + items[3] + "," + user_str + "," + user_time_str , file=file_output)
manual_join()
# -----------------------------------------taobao_last_samples-----------------------------------------
def create_last_examples():  # 添加训练集还有(测试集或验证集)的标志（前边加2018或者2019）
    fi = open(f"../dataset/taobao/taobao_sorted_with_neg", "r")  # 0,A2G5TCU2WDFZ65,B0079NXLM0,Carry-Ons,default_user,-1,1321574400
    fo = open(f"../dataset/taobao/taobao_last_samples", "w")  # 1,A10000012B7CGYKOMPQ4L,000100039X,Clothing,1355616000
    # 统计每个用户的行为数目，包括负样本(实际上这种统计信息也可以作为特征输入神经网络)
    user_count = {}
    for line in fi:
        user = line.strip().split(",")[1]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    fi.seek(0)

    i = 0
    last_user = "-1"
    num_line = sum([1 for l in open(f"../dataset/taobao/taobao_sorted_with_neg", "r")])
    for line in tqdm(fi, total=num_line):
        line = line.strip()
        user = line.split(",")[1]
        if user == last_user:  # 引入last_user是为了确定是否重置计数变量i，遍历user_count[user]中每一个行为
            if i < user_count[user] - 4:
                print("train" + "," + line, file=fo)  # 剩余的训练集
            elif i < user_count[user] - 2:
                print("valid" + "," + line, file=fo)  # 我们用的训练集
            else:
                print("test" + "," + line, file=fo)  # 验证集和测试集
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 4:
                print("train" + "," + line, file=fo)  # 剩余的训练集
            elif i < user_count[user] - 2:
                print("valid" + "," + line, file=fo)  # 我们用的训练集
            else:
                print("test" + "," + line, file=fo)  # 验证集和测试集
        i += 1

# -----------------------------------------taobao_last_samples-----------------------------------------
def split_train_and_test_before_V1():
    fin = open(r"../dataset/taobao/taobao_last_samples", "r")
    ftrain = open(r"../dataset/taobao/taobao_train_V1", "w")
    ftest = open(r"../dataset/taobao/taobao_test_before_V1", "w")
    gap = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])  # 时间间隔
    last_user = "-1"
    num_line = sum([1 for i in open("../dataset/taobao/taobao_last_samples", "r")])
    for line in tqdm(fin, total=num_line):  # 20180117,0,207129,3829634,2355072,1511591442
        fo = None
        items = line.strip().split(",")
        train_or_test = items[0]  # 决定分类到训练集还是测试集2018/2019
        click = int(items[1])  # 是1的话才是正样本，用户交互过的行为，int是将字符串转换成数字
        user = items[2]
        item_id = items[3]
        item_cat = items[4]
        target_time = items[5]  # 时间，没有[4]是因为评分不重要：正负样本不是靠评分划分的，只用clk即可

        if train_or_test == "20180118":
            fo = ftrain
        elif train_or_test == "20180119":
            fo = ftest

        if user != last_user:  # 如果是一个新的用户，就新开三个列表，存储用户的历史行为（物品id，类别，间隔）
            item_id_list = []
            item_cat_lsit = []
            history_time_list = []
            flag=False
        else:
            # 累积的历史行为
            history_clk_num = len(item_id_list)
            mid_id = []
            cat_id = []
            tiv_target_id = []
            tiv_neighbor_id = []
            for mid in item_id_list:
                mid_id.append(mid)
            for cat in item_cat_lsit:
                cat_id.append(cat)
            for time in history_time_list:
                tiv = float(target_time) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0  # 间隔多少天,amazon时间戳只精确到天
                tiv_target_id.append(str(np.sum(tiv >= gap)))  # 用户的行为间隔
            for i in range(len(history_time_list)):
                if i <= 0: continue
                last_time = history_time_list[i - 1]
                temp_time = history_time_list[i]
                tiv1 = float(temp_time) / 3600.0 / 24.0 - float(last_time) / 3600.0 / 24.0  # 行为物品之间间隔间隔多少天（此处我们没有计算第一个行为物品：因为其之前没有用户行为），但是在后边embedding的时候，我们可以赋予一个默认值
                tiv_neighbor_id.append(str(np.sum(tiv1 >= gap)))  # 用户的行为间隔
            if len(history_time_list) > 0:
                tiv1 = float(target_time) / 3600.0 / 24.0 - float(history_time_list[-1]) / 3600.0 / 24.0  # 目标物品与最后的行为物品间隔多少天
                tiv_neighbor_id.append(str(np.sum(tiv1 >= gap)))

            if (history_clk_num > 50):  # 这里需要截断，因为这里文件不截断的话训练的时候需要截断，更浪费时间
                mid_id = mid_id[history_clk_num - 50:]
                cat_id = cat_id[history_clk_num - 50:]
                tiv_target_id = tiv_target_id[history_clk_num - 50:]
                tiv_neighbor_id = tiv_neighbor_id[history_clk_num - 50:]
            mid_str = "".join(mid_id)
            cat_str = "".join(cat_id)
            history_time_str = "".join(history_time_list)
            tiv_target_str = "".join(tiv_target_id)
            tiv_neighbor_str = "".join(tiv_neighbor_id)

            if fo != None and history_clk_num >= 3 :  # 8 is the average length of user behavior
                # item_id是目标物品id，item_cat是目标物品类别，mid_str，cat_str是历史行为
                if fo==ftrain:
                    flag=True
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str, file=fo)
                if fo==ftest and flag:
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str, file=fo)

        last_user = user
        if click:  # 只有点击了才算进历史行为
            item_id_list.append(item_id)
            item_cat_lsit.append(item_cat)
            history_time_list.append(target_time)

def part_of_test_to_train():
    fi = open("../dataset/taobao/taobao_test_before_V1", "r")
    ftrain = open("../dataset/taobao/taobao_train_V1", "a")
    ftest = open("../dataset/taobao/taobao_test_V1", "w")
    fvalid = open("../dataset/Electronics/taobao_valid_V2", "w")

    # 是随机从1~10中选取整数，如果恰好是2，当前用户3就作为验证数据集
    while True:
        rand_int = random.randint(1, 10)
        # 连续读两行
        no_clk_line = fi.readline().strip()  # 读一行
        clk_line = fi.readline().strip()  # 再读一行
        if no_clk_line == "" or clk_line == "":  # 保证正负样本成对存在
            break
        if rand_int == 1 or rand_int == 8:
            print(no_clk_line, file=ftest)
            print(clk_line, file=ftest)
        elif rand_int == 2 or rand_int == 9:
            print(no_clk_line, file=fvalid)
            print(clk_line, file=fvalid)
        else:
            print(no_clk_line, file=ftrain)
            print(clk_line, file=ftrain)


def generate_voc():
    f_train = open("../dataset/taobao/taobao_train", "r")  # 1,A13QGAPPBC1PVS,B00E3V3CAW,Casual,1402704000,B007KR07ZYB005NKKJUSB00CQ3DJGQ,Knits & TeesArm WarmersCasual,10109
    uid_dict = {}
    mid_dict = {}
    cat_dict = {}
    tiv_dict = {}

    for line in f_train:
        arr = line.strip("\n").split(",")
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[5]
        cat_list = arr[6]
        tiv_list = arr[7]
        if uid not in uid_dict:
            uid_dict[uid] = 0
        uid_dict[uid] += 1
        if mid not in mid_dict:
            mid_dict[mid] = 0
        mid_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(""):
            if m not in mid_dict:
                mid_dict[m] = 0
            mid_dict[m] += 1
        for c in cat_list.split(""):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1
        for tiv in tiv_list.split(""):
            if tiv not in tiv_dict:
                tiv_dict[tiv] = 0
            tiv_dict[tiv] += 1

    sorted_uid_dict = sorted(uid_dict.items(), key=lambda x: x[1], reverse=True)  # 按照出现的次数排序
    sorted_mid_dict = sorted(mid_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_tiv_dict = sorted(tiv_dict.items(), key=lambda x: x[1], reverse=True)

    uid_voc = {}
    uid_voc["default_mid"] = 0
    index = 1
    for key, value in sorted_uid_dict:
        uid_voc[key] = index
        index += 1

    mid_voc = {}
    mid_voc["default_mid"] = 0
    index = 1
    for key, value in sorted_mid_dict:
        mid_voc[key] = index
        index += 1

    cat_voc = {}
    cat_voc["default_cat"] = 0
    index = 1
    for key, value in sorted_cat_dict:
        cat_voc[key] = index
        index += 1

    tiv_voc = {}
    index = 0  # 目标物品的时间间隔
    for key, value in sorted_tiv_dict:
        tiv_voc[key] = index
        index += 1

    pickle.dump(uid_voc, open("../dataset/taobao/uid_voc.pkl", "wb+"))  # 二进制文件
    pickle.dump(mid_voc, open("../dataset/taobao/mid_voc.pkl", "wb+"))
    pickle.dump(cat_voc, open("../dataset/taobao/cat_voc.pkl", "wb+"))
    pickle.dump(tiv_voc, open("../dataset/taobao/tiv_voc.pkl", "wb+"))  # process_meta('meta_taobao.json')#0000031887,Active Skirts


# create_iteminfo()
# process_reviews('reviews_taobao_5.json')#A1KLRMWW2FWPL4,0000031887,1297468800
# manual_join()  # split_all_example()
# split_train_test()
# part_of_test_to_train_split_by_user()
# generate_voc()
