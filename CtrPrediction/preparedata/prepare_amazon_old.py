# -*- coding: UTF-8 -*-
'''
如果按照一个用户只有一个正样本来划分的话，过拟合现象会很严重。因为训练的样本太少
'''

import json
import pickle
import random

import numpy as np
from tqdm import tqdm

np.random.seed(1234)
random.seed(1234)
file_name = 'Clothing'


# -----------------------------------------item-info-----------------------------------------
# 从物品元数据json中得到asin，categories两列，保存到item-info中
def process_meta():
    file_input = open(f'../dataset/{file_name}_new/meta_{file_name}.json', "r", encoding="utf-8")
    file_output = open("../dataset/{file_name}_new/item-info", "w", encoding="utf-8")
    for line in file_input:
        obj = json.loads(line)  # load是加载文件，loads（s是string）是解析字符串
        cat = obj["category"][1] if len(obj["category"]) > 1 else "default"  # 2018版的数据集用这个
        # cat = obj["categories"][0][-1]  # 2014版的数据集，选择最细粒度的类别。'categories': [['Clothing', 'Computers & Accessories', 'Cables & Accessories', 'Monitor Accessories']]
        print(obj["asin"] + "," + cat, file=file_output)


# -----------------------------------------reviews-info-----------------------------------------
# 因为亚马逊评论里有较多其它的内容，我们这里仅提取出我们有用的[用户id，物品id，时间戳]
def process_reviews():
    file_input = open(f'../dataset/{file_name}/reviews_{file_name}_5.json', "r")
    file_output = open(f"../dataset/{file_name}/reviews-info", "w")
    for line in file_input:
        obj = json.loads(line)  # 每一行都是json形式
        userID = obj["reviewerID"]
        itemID = obj["asin"]
        time = obj["unixReviewTime"]
        print(userID + "," + itemID + "," + str(time), file=file_output)


# -----------------------------------------Clothing_sorted_with_neg-----------------------------------------
# 这里已经按用户排好序了，并且考虑了物品的行为序列（是否点击，用户id，物品id，物品类别，购买的时间戳，物品行为序列（在该时间之前购买该物品的用户，在该时间之前购买该物品的用户的购买时间）），但是没有考虑时间间隔
# 1,A1KLRMWW2FWPL4,B00400N6XE,Shoes,1353542400,A2TB8EVUMJQJODA1P99HOPB4LB9IAMMV9GX0MEVOBA184ZFZZBWEXMSA1EZJSN08SOB5L,1292889600.01297382400.01324771200.01326326400.01328572800.0
# 这里仅仅是构造用户每一次的行为（正负），但是未合成序列{'A1KLRMWW2FWPL4': [(['A1KLRMWW2FWPL4', '0000031887', '1297468800'], 1297468800.0), (['A1KLRMWW2FWPL4', 'B000FH4JJQ', '1368576000'], 1368576000.0)}
def manual_join():
    with open(f"../dataset/{file_name}/reviews-info", "r") as file_review:
        user_map = {}
        item_list = []  # 这里item_list是为了记录所有出现过的物品（可重复）供后面随机抽负样本用
        for line in file_review:
            items = line.strip().split(",")  # [reviewerID,asin,unixReviewTime]
            if items[0] not in user_map:
                user_map[items[0]] = []
            user_map[items[0]].append((items, float(items[-1])))
            item_list.append(items[1])
    # 对用户的行为序列进行排序
    sorted_item_in_user_seq = {key: sorted(value, key=lambda x: x[-1]) for key, value in user_map.items()}
    # 构造物品行为序列，曾经购买过该物品的用户 :[用户，时间]{'0000031887': [('A1KLRMWW2FWPL4', 1297468800.0), ('A2G5TCU2WDFZ65', 1358553600.0)]}
    file_review = open(f"../dataset/{file_name}/reviews-info", "r")
    useridToClickItem = {}
    for line in file_review:
        items = line.strip().split(",")  # A1KLRMWW2FWPL4	0000031887	5.0	1297468800
        if items[1] not in useridToClickItem:
            useridToClickItem[items[1]] = []
        useridToClickItem[items[1]].append((items[0], float(items[-1])))  # data[0]是用户，data[-1]是时间
    # 对物品的行为序列进行排序
    useridToClickItem = {key: sorted(value, key=lambda x: x[1]) for key, value in useridToClickItem.items()}
    # item_map：（itemID，categoryID）
    file_meta = open(f"../dataset/{file_name}/item-info", "r")
    item_map = {}
    for line in file_meta:
        arr = line.strip().split(",")
        if arr[0] not in item_map:
            item_map[arr[0]] = arr[1]

    file_output = open(f"../dataset/{file_name}/{file_name}_sorted_with_neg1", "w")
    # gap = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    for key, value in sorted_item_in_user_seq.items():  # 遍历所有用户
        for items, _ in value:  # 对该用户的每个行为构造负样本  # A1KLRMWW2FWPL4	0000031887	1297468800
            asin = items[1]
            # days = float(items[2]) // 3600 // 24  # 从1970年1月1日到购买该物品时的天数
            negative_num = 0
            # 负样本的构造
            while True:
                asin_neg_index = random.randint(0, len(item_list) - 1)
                asin_neg = item_list[asin_neg_index]
                if asin_neg == asin:
                    continue
                # 去判断一下有没有在当前时间之前购买的用户，算作物品的历史行为(也有可能该用户是第一个购买的，所以物品的历史行为仍然为0，下面还要处理)
                user_count = 0
                user_str = ""
                user_time_str = ""
                for (user, time) in useridToClickItem[asin_neg]:
                    if (int(time) > int(items[-1])) or user_count > 5:  # 如果找到一个用户购买该物品的时间在该用户购买该物品的时间之后，那么后边的都不是物品行为序列了（因为时间超了）
                        break  # 直接break的原因就是已经排好序，后边的物品肯定也满足条件
                    if user == items[0]:  # 这个用户购买的不算，也就是说该用户有可能已经购买过该物品？但是这里不算之前购买过的？
                        continue
                    user_str += user + ""
                    user_time_str += str(time) + ""
                    user_count += 1  # user_tiv = float(days) - time // 3600 // 24 + 1.  # 当前点击时间与之前点击时间相差的天数  # user_tiv_str += str(np.sum(user_tiv >= gap)) + ""  # [ True  True  True  True  True  True False False False False False False]->6

                if len(user_str) > 0:
                    user_str = user_str[:-1]  # 去掉最后面的""
                    user_time_str = user_time_str[:-1]  # 去掉最后面的""
                if len(user_str) == 0:  # 如果该物品在此用户购买之前没有用户购买过，也就是该用户是第一个购买该物品的用户，那么赋予一个默认的用户，保证文件中此项不为空
                    user_str = "default_user"
                    user_time_str = "-1"

                if asin_neg in item_map:
                    print("0" + "," + items[0] + "," + asin_neg + "," + item_map[asin_neg] + "," + items[2] + "," + user_str + "," + user_time_str, file=file_output)
                else:
                    print("0" + "," + items[0] + "," + asin_neg + "," + "default_cat" + "," + items[2] + "," + user_str + "," + user_time_str, file=file_output)
                negative_num += 1
                if negative_num == 1:  # 负样本的数目
                    break
            # 正样本
            user_count = 0
            user_str = ""
            user_time_str = ""
            for (user, time) in useridToClickItem[asin]:
                if int(time) > int(items[-1]) or user_count > 5:
                    break
                if user == items[0]:
                    continue
                user_str += user + ""  # 去掉最后面的""
                user_time_str += str(time) + ""  # 去掉最后面的""
                user_count += 1  # user_tiv = float(days) - time // 3600 // 24 + 1.  # user_time_str += str(np.sum(user_tiv >= gap)) + ""
            if len(user_str) > 0:
                user_str = user_str[:-1]
                user_time_str = user_time_str[:-1]
            if len(user_str) == 0:
                user_str = "default_user"
                user_time_str = "-1"
            if asin in item_map:  # 用户id，目标物品id，目标物品类别id，购买目标物品时间，在此时间之前购买过该物品的用户，在此时间之前购买过该物品的用户的购买时间
                print("1" + "," + items[0] + "," + items[1] + "," + item_map[asin] + "," + items[2] + "," + user_str + "," + user_time_str, file=file_output)
            else:
                print("1" + "," + items[0] + "," + items[1] + "," + "default_cat" + "," + items[2] + "," + user_str + "," + user_time_str, file=file_output)


manual_join()


# -----------------------------------------Clothing_last_samples-----------------------------------------
# 就是加上前缀：train，valid，test
# 2017/2018/2019，clk，用户，物品，类别，时间戳
def create_last_examples():  # 添加训练集还有(测试集或验证集)的标志（前边加2018或者2019）
    fi = open(f"../dataset/{file_name}/{file_name}_sorted_with_neg", "r")  # 0,A2G5TCU2WDFZ65,B0079NXLM0,Carry-Ons,default_user,-1,1321574400
    fo = open(f"../dataset/{file_name}/{file_name}_last_samples", "w")  # 1,A10000012B7CGYKOMPQ4L,000100039X,Clothing,1355616000
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
    num_line = sum([1 for l in open(f"../dataset/{file_name}/{file_name}_sorted_with_neg", "r")])
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


# 这里是不管用户行为数据的多少，一个用户都对应一对训练样本（正负样本），原因是为了减少训练样本的数量进而减小开销（或者另一个原因就是只选取最近的行为预测）。
# 但是个人觉得其实应该用户行为数据多的对应更多的训练样本

# 这里是不管用户行为数据的多少，一个用户都对应一对训练样本（正负样本）。并且有相邻的时间间隔
def split_train_and_test_before_V1():
    fin = open(f"../dataset/{file_name}/{file_name}_last_samples", "r")  # train,0,A1KLRMWW2FWPL4,B009RIZWYK,Shoes & Accessories: International Shipping Available,default_user,-1,1297468800
    ftrain = open(f"../dataset/{file_name}/{file_name}_train_V1", "w")  # 用户个数乘以2（一个用户对应一个正样本，一个负样本）
    ftest = open(f"../dataset/{file_name}/{file_name}_test_before_V1", "w")
    gap = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])  # 时间间隔，亚马逊只精确到天级别。（其实远远弱于会话级别）
    last_user = "-1"
    num_line = sum([1 for i in open(f"../dataset/{file_name}/{file_name}_last_samples", "r")])
    for line in tqdm(fin, total=num_line):  # train,0,A10000012B7CGYKOMPQ4L,1939927196,Clothing,1355616000
        fo = None
        items = line.strip().split(",")
        train_or_test = items[0]  # 决定分类到训练集还是测试集2018/2019
        click = int(items[1])  # 是1的话才是正样本，用户交互过的行为，int是将字符串转换成数字
        user = items[2]
        item_id = items[3]
        item_cat = items[4]
        target_time = items[5]  # 时间

        if train_or_test == "valid":
            fo = ftrain
        elif train_or_test == "test":
            fo = ftest

        if user != last_user:  # 如果是一个新的用户，就新开三个列表，存储用户的历史行为（物品id，类别，间隔）
            item_id_list = []
            history_item_cat_list = []
            history_item_time_list = []
            flag = False  # 观察一个新用户，是否被选入训练集；比如一个用户的历史行为长4，我们又要求最低长度为4，这时候test才符合要求，那么会进入test，但是此时train里面没有该用户；这样的例子我们排除掉
        else:
            # 累积的历史行为
            history_clk_num = len(item_id_list)
            mid_id = []
            cat_id = []
            tiv_to_target_list = []
            tiv_to_neighbor_list = []
            for mid in item_id_list:
                mid_id.append(mid)
            for cat in history_item_cat_list:
                cat_id.append(cat)
            # -------------------------计算历史行为物品与目标物品的时间间隔--------------------------------
            for time in history_item_time_list:
                tiv = float(target_time) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0  # 间隔多少天,amazon时间戳只精确到天
                tiv_to_target_list.append(str(np.sum(tiv >= gap)))  # 用户的行为间隔，离散化
            # -------------------------计算相邻的时间间隔，算上目标物品--------------------------------
            for i in range(len(history_item_time_list)):
                if i <= 0: continue
                last_time = history_item_time_list[i - 1]
                temp_time = history_item_time_list[i]
                tiv1 = float(temp_time) / 3600.0 / 24.0 - float(last_time) / 3600.0 / 24.0  # 行为物品之间间隔间隔多少天（此处我们没有计算第一个行为物品：因为其之前没有用户行为），但是在后边embedding的时候，我们可以赋予一个默认值
                tiv_to_neighbor_list.append(str(np.sum(tiv1 >= gap)))  # 用户的行为间隔
            if len(history_item_time_list) > 0:
                tiv1 = float(target_time) / 3600.0 / 24.0 - float(history_item_time_list[-1]) / 3600.0 / 24.0  # 目标物品与最后的行为物品间隔多少天
                tiv_to_neighbor_list.append(str(np.sum(tiv1 >= gap)))
            # -------------------------这里需要截断，因为这里文件不截断的话训练的时候需要截断，更浪费时间--------------------------------
            if (history_clk_num > 50):
                mid_id = mid_id[history_clk_num - 50:]
                cat_id = cat_id[history_clk_num - 50:]
                tiv_to_target_list = tiv_to_target_list[history_clk_num - 50:]
                tiv_to_neighbor_list = tiv_to_neighbor_list[history_clk_num - 50:]
            mid_str = "".join(mid_id)
            cat_str = "".join(cat_id)
            history_time_str = "".join(history_item_time_list)
            tiv_target_str = "".join(tiv_to_target_list)
            tiv_neighbor_str = "".join(tiv_to_neighbor_list)

            if fo != None and history_clk_num >= 3:  # 8 is the average length of user behavior
                # item_id是目标物品id，item_cat是目标物品类别，mid_str，cat_str是历史行为
                if fo == ftrain:
                    flag = True  # 这里的flag是说该用户也存在于训练集中，只有在训练集中出现过的才会被选进测试集进行测试，否则没有训练过的不进行测试
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str, file=fo)
                if fo == ftest and flag:
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str, file=fo)

        last_user = user
        if click:  # 只有点击了才算进历史行为
            item_id_list.append(item_id)
            history_item_cat_list.append(item_cat)
            history_item_time_list.append(target_time)


# 用户行为数据多的对应更多的训练样本,并且有相邻的时间间隔
# click_label, user, target_item_id, target_item_cat, target_time, history_item_ids, history_cat_ids, history_time,tiv_target_str,tiv_neighbor_str
def split_train_and_test_before_V2():
    fin = open(f"../dataset/{file_name}/{file_name}_last_samples", "r")  # train,1,A1KLRMWW2FWPL4,B003U3GOFO,Clothing,1360281600
    ftrain = open(f"../dataset/{file_name}/{file_name}_train_V2", "w")
    ftest = open(f"../dataset/{file_name}/{file_name}_test_before_V2", "w")
    gap = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])  # 时间间隔
    last_user = "-1"
    num_line = sum([1 for l in open("../dataset/Clothing/Clothing_last_samples", "r")])
    for line in tqdm(fin, total=num_line):  # train,0,A10000012B7CGYKOMPQ4L,1939927196,Clothing,1355616000
        fo = ftrain
        items = line.strip().split(",")
        train_or_test = items[0]  # 决定分类到训练集还是测试集train/valid/test
        click = int(items[1])  # 是1的话才是正样本，用户交互过的行为，int是将字符串转换成数字
        user = items[2]
        item_id = items[3]
        item_cat = items[4]
        target_time = items[5]  # 时间戳

        if train_or_test == "test":
            fo = ftest

        if user != last_user:  # 如果是一个新的用户，就新开三个列表，存储用户的历史行为（物品id，类别，间隔）
            item_id_list = []
            history_item_cat_list = []
            history_item_time_list = []
            flag = False  # 观察一个新用户，是否被选入训练集；比如一个用户的历史行为长4，我们又要求最低长度为4，这时候test才符合要求，那么会进入test，但是此时train里面没有该用户；这样的例子我们排除掉
        else:
            # 累积的历史行为
            history_clk_num = len(item_id_list)
            mid_id = []
            cat_id = []
            tiv_to_target_list = []
            tiv_to_neighbor_list = []
            for mid in item_id_list:
                mid_id.append(mid)
            for cat in history_item_cat_list:
                cat_id.append(cat)
            # -------------------------计算历史行为物品与目标物品的时间间隔--------------------------------
            for time in history_item_time_list:
                tiv = float(target_time) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0  # 间隔多少天,amazon时间戳只精确到天
                tiv_to_target_list.append(str(np.sum(tiv >= gap)))  # 用户的行为间隔，离散化
            # -------------------------计算相邻的时间间隔，算上目标物品--------------------------------
            for i in range(len(history_item_time_list)):
                if i <= 0: continue
                last_time = history_item_time_list[i - 1]
                temp_time = history_item_time_list[i]
                tiv1 = float(temp_time) / 3600.0 / 24.0 - float(last_time) / 3600.0 / 24.0  # 行为物品之间间隔间隔多少天（此处我们没有计算第一个行为物品：因为其之前没有用户行为），但是在后边embedding的时候，我们可以赋予一个默认值
                tiv_to_neighbor_list.append(str(np.sum(tiv1 >= gap)))  # 用户的行为间隔
            if len(history_item_time_list) > 0:
                tiv1 = float(target_time) / 3600.0 / 24.0 - float(history_item_time_list[-1]) / 3600.0 / 24.0  # 目标物品与最后的行为物品间隔多少天
                tiv_to_neighbor_list.append(str(np.sum(tiv1 >= gap)))
            # -------------------------这里需要截断，因为这里文件不截断的话训练的时候需要截断，更浪费时间--------------------------------
            if (history_clk_num > 50):
                mid_id = mid_id[history_clk_num - 50:]
                cat_id = cat_id[history_clk_num - 50:]
                tiv_to_target_list = tiv_to_target_list[history_clk_num - 50:]
                tiv_to_neighbor_list = tiv_to_neighbor_list[history_clk_num - 50:]
            mid_str = "".join(mid_id)
            cat_str = "".join(cat_id)
            history_time_str = "".join(history_item_time_list)
            tiv_target_str = "".join(tiv_to_target_list)
            tiv_neighbor_str = "".join(tiv_to_neighbor_list)

            if fo != None and history_clk_num >= 3:  # 8 is the average length of user behavior
                # item_id是目标物品id，item_cat是目标物品类别，mid_str，cat_str是历史行为
                if fo == ftrain:
                    flag = True  # 这里的flag是说只有在训练集中出现过的才会被选进测试集
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str, file=fo)
                if fo == ftest and flag:
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str, file=fo)
        last_user = user
        if click:  # 只有点击了才算进历史行为
            item_id_list.append(item_id)
            history_item_cat_list.append(item_cat)
            history_item_time_list.append(target_time)


# 这里包括物品行为，对于books等较大的数据集，要使fo=None，否则会太大
# 可以将该文件拆分成多个小文件，一起执行，最后合并，要不然执行得太慢了
def split_train_and_test_before_V3():
    fin = open(f"../dataset/{file_name}/{file_name}_last_samples", "r")  # train,1,A1KLRMWW2FWPL4,0000031887,Active Skirts,default_user,-1,1297468800
    ftrain = open(f"../dataset/{file_name}/{file_name}_train_V3", "w")
    ftest = open(f"../dataset/{file_name}/{file_name}_test_before_V3", "w")
    gap = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])  # 时间间隔
    last_user = "-1"
    num_line = sum([1 for l in open(f"../dataset/{file_name}/{file_name}_last_samples", "r")])
    # 遍历整个文件构造训练和测试样本
    for line in tqdm(fin, total=num_line):  # train,0,A10000012B7CGYKOMPQ4L,1939927196,Clothing,1355616000
        fo = None
        items = line.strip().split(",")
        train_or_test = items[0]  # 决定分类到训练集还是测试集train/valid/test
        click = int(items[1])  # 是1的话才是正样本，用户交互过的行为，int是将字符串转换成数字
        user = items[2]
        item_id = items[3]
        item_cat = items[4]
        target_time = items[5]  # 时间戳
        user_str = items[6]  # 在此时刻之前同样购买过该物品的历史用户
        user_time_str = items[7]  # 在此时刻之前同样购买过该物品的历史用户购买的时间
        if train_or_test == "valid":
            fo = ftrain
        elif train_or_test == "test":
            fo = ftest
        if user != last_user:  # 如果是一个新的用户，就新开5个列表，用于存储当前用户的历史行为：历史物品id，历史物品类别，历史物品距当前目标物品的时间间隔，历史物品的行为，历史物品的行为时间
            history_item_id_list = []
            history_item_cat_list = []
            history_item_time_list = []
            history_items_behaviors_list = []  # 元素是一个字符串：A2U0LY01LK9C3EA3P0CEE0MD7MAYA33PLZ7SD5MCG0A3PD8JD9L4WEII
            history_items_behaviors_time_list = []  # 元素是一个字符串：1319587200.01321747200.01325289600.01328659200.0
            flag = False
        elif fo != None:  # 如果fo==None的话，只需要加历史行为就行了，其它的都不用计算
            # 对历史行为做一些计算（需要截断，需要将时间转换成时间间隔），生成需要的样本格式存入文件
            mid_id = []  # 对应history_item_id_list
            cat_id = []  # 对应history_item_cat_list
            tiv_to_target_list = []  # 对应history_item_time_list
            tiv_to_neighbor_list = []  # 对应history_item_time_list

            tmp_history_items_behaviors_list = []  # 对应history_items_behaviors_list
            tmp_history_items_behaviors_time_list = []  # 对应history_items_behaviors_time_list

            for mid in history_item_id_list:
                mid_id.append(mid)
            for cat in history_item_cat_list:
                cat_id.append(cat)
            for item_behaviors in history_items_behaviors_list:
                tmp_list = item_behaviors.split('')
                tmp_history_items_behaviors_list.append(''.join(tmp_list[-3:]))  # 每个历史物品的行为取后面三个
            # -------------------------计算历史行为物品与目标物品的时间间隔，算上目标物品--------------------------------
            for time in history_item_time_list:
                tiv = float(target_time) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0  # 间隔多少天,amazon时间戳只精确到天
                tiv_to_target_list.append(str(np.sum(tiv >= gap)))  # 用户的行为间隔，离散分桶化
            # -------------------------计算相邻的时间间隔，算上目标物品--------------------------------
            for i in range(len(history_item_time_list)):
                if i <= 0: continue  # 此处不计算第一个行为物品，因为之前该用户没有行为
                tiv = float(history_item_time_list[i]) / 3600.0 / 24.0 - float(history_item_time_list[i - 1]) / 3600.0 / 24.0  # 行为物品之间间隔间隔多少天（此处我们没有计算第一个行为物品：因为其之前没有用户行为），但是在后边embedding的时候，我们可以赋予一个默认值
                tiv_to_neighbor_list.append(str(np.sum(tiv >= gap)))  # 用户的行为间隔
            if len(history_item_time_list) > 0:
                tiv = float(target_time) / 3600.0 / 24.0 - float(history_item_time_list[-1]) / 3600.0 / 24.0  # 目标物品与最后的行为物品间隔多少天
                tiv_to_neighbor_list.append(str(np.sum(tiv >= gap)))
            # -------------------------历史物品的行为列表--------------------------------
            for i in range(len(history_item_time_list)):  # 实际上这里的长度跟item_id_list的长度是一样的
                now_user_time = history_item_time_list[i]  # 获取当前用户购买历史物品的时间
                last_users_time = history_items_behaviors_time_list[i].split('')  # 得到在当前用户之前的用户购买该物品的时间，这是一个包含多个时间的字符串，我们将其转换为了列表方便计算时间间隔
                tmp_list = []
                for j in range(len(last_users_time)):  # 得到每一个历史物品的行为序列的时间间隔
                    tiv = float(now_user_time) / 3600.0 / 24.0 - float(last_users_time[j]) / 3600.0 / 24.0
                    tmp_list.append(str(np.sum(tiv >= gap)))
                tmp_str = ''.join(tmp_list[-3:])
                tmp_history_items_behaviors_time_list.append(tmp_str)

            # -------------------------这里需要截断，因为这里文件不截断的话训练的时候需要截断，更浪费时间--------------------------------
            history_clk_num = len(history_item_id_list)
            if (history_clk_num > 50):
                mid_id = mid_id[history_clk_num - 50:]
                cat_id = cat_id[history_clk_num - 50:]
                tiv_to_target_list = tiv_to_target_list[history_clk_num - 50:]
                tiv_to_neighbor_list = tiv_to_neighbor_list[history_clk_num - 50:]
                tmp_history_items_behaviors_list = tmp_history_items_behaviors_list[history_clk_num - 50:]
                tmp_history_items_behaviors_time_list = tmp_history_items_behaviors_time_list[history_clk_num - 50:]
            # 下面是将列表转换成字符串存入文件
            mid_str = "".join(mid_id)
            cat_str = "".join(cat_id)
            history_time_str = "".join(history_item_time_list)
            tiv_target_str = "".join(tiv_to_target_list)
            tiv_neighbor_str = "".join(tiv_to_neighbor_list)
            history_items_behaviors_str = "#".join(tmp_history_items_behaviors_list)  # 这里最多只留三个最靠近的
            history_items_behaviors_time_str = "#".join(tmp_history_items_behaviors_time_list)

            if history_clk_num >= 3:  # 8 is the average length of user behavior
                # item_id是目标物品id，item_cat是目标物品类别，mid_str，cat_str是历史行为
                if fo == ftrain:
                    flag = True  # 这里的flag是说只有在训练集中出现过的才会被选进测试集
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str + "," + history_items_behaviors_str + "," + history_items_behaviors_time_str, file=fo)
                if fo == ftest and flag:
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str + "," + history_items_behaviors_str + "," + history_items_behaviors_time_str, file=fo)
        last_user = user
        # 累积历史行为
        if click:  # 只有点击了才算进历史行为
            history_item_id_list.append(item_id)
            history_item_cat_list.append(item_cat)
            history_item_time_list.append(target_time)
            history_items_behaviors_list.append(user_str)
            history_items_behaviors_time_list.append(user_time_str)


def split_train_and_test_before_V4():
    fin = open(f"../dataset/{file_name}/{file_name}_last_samples", "r")  # train,1,A1KLRMWW2FWPL4,0000031887,Active Skirts,default_user,-1,1297468800
    ftrain = open(f"../dataset/{file_name}/{file_name}_train_V4", "w")
    ftest = open(f"../dataset/{file_name}/{file_name}_test_before_V4", "w")
    gap = np.array([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])  # 时间间隔

    last_user = "-1"
    num_line = sum([1 for l in open(f"../dataset/{file_name}/{file_name}_last_samples", "r")])
    # 遍历整个文件构造训练和测试样本
    for line in tqdm(fin, total=num_line):  # train,0,A10000012B7CGYKOMPQ4L,1939927196,Clothing,1355616000
        fo = ftrain
        items = line.strip().split(",")
        train_or_test = items[0]  # 决定分类到训练集还是测试集train/valid/test
        click = int(items[1])  # 是1的话才是正样本，用户交互过的行为，int是将字符串转换成数字
        user = items[2]
        item_id = items[3]
        item_cat = items[4]
        target_time = items[5]  # 时间戳
        user_str = items[6]  # 在此时刻之前同样购买过该物品的历史用户
        user_time_str = items[7]  # 在此时刻之前同样购买过该物品的历史用户购买的时间

        if train_or_test == "test":
            fo = ftest
        if user != last_user:  # 如果是一个新的用户，就新开5个列表，用于存储当前用户的历史行为：历史物品id，历史物品类别，历史物品距当前目标物品的时间间隔，历史物品的行为，历史物品的行为时间
            history_item_id_list = []
            history_item_cat_list = []
            history_item_time_list = []

            history_items_behaviors_list = []  # 元素是一个字符串：A2U0LY01LK9C3EA3P0CEE0MD7MAYA33PLZ7SD5MCG0A3PD8JD9L4WEII
            history_items_behaviors_time_list = []  # 元素是一个字符串：1319587200.01321747200.01325289600.01328659200.0
            flag = False
        elif fo != None:  # 如果fo==None的话，只需要加历史行为就行了，其它的都不用计算
            # 对历史行为做一些计算（需要截断，需要将时间转换成时间间隔），生成需要的样本格式存入文件
            mid_id = []  # 对应history_item_id_list
            cat_id = []  # 对应history_item_cat_list
            tiv_to_target_list = []  # 对应history_item_time_list
            tiv_to_neighbor_list = []  # 对应history_item_time_list

            tmp_history_items_behaviors_list = []  # 对应history_items_behaviors_list
            tmp_history_items_behaviors_time_list = []  # 对应history_items_behaviors_time_list

            for mid in history_item_id_list:
                mid_id.append(mid)
            for cat in history_item_cat_list:
                cat_id.append(cat)
            for item_behaviors in history_items_behaviors_list:
                tmp_list = item_behaviors.split('')
                tmp_history_items_behaviors_list.append(''.join(tmp_list[-3:]))  # 每个历史物品的行为取后面三个
            # -------------------------计算历史行为物品与目标物品的时间间隔，算上目标物品--------------------------------
            for time in history_item_time_list:
                tiv = float(target_time) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0  # 间隔多少天,amazon时间戳只精确到天
                tiv_to_target_list.append(str(np.sum(tiv >= gap)))  # 用户的行为间隔，离散分桶化
            # -------------------------计算相邻的时间间隔，算上目标物品--------------------------------
            for i in range(len(history_item_time_list)):
                if i <= 0: continue  # 此处不计算第一个行为物品，因为之前该用户没有行为
                tiv = float(history_item_time_list[i]) / 3600.0 / 24.0 - float(history_item_time_list[i - 1]) / 3600.0 / 24.0  # 行为物品之间间隔间隔多少天（此处我们没有计算第一个行为物品：因为其之前没有用户行为），但是在后边embedding的时候，我们可以赋予一个默认值
                tiv_to_neighbor_list.append(str(np.sum(tiv >= gap)))  # 用户的行为间隔
            if len(history_item_time_list) > 0:
                tiv = float(target_time) / 3600.0 / 24.0 - float(history_item_time_list[-1]) / 3600.0 / 24.0  # 目标物品与最后的行为物品间隔多少天
                tiv_to_neighbor_list.append(str(np.sum(tiv >= gap)))
            # -------------------------历史物品的行为列表--------------------------------
            for i in range(len(history_item_time_list)):  # 实际上这里的长度跟item_id_list的长度是一样的
                now_user_time = history_item_time_list[i]  # 获取当前用户购买历史物品的时间
                last_users_time = history_items_behaviors_time_list[i].split('')  # 得到在当前用户之前的用户购买该物品的时间，这是一个包含多个时间的字符串，我们将其转换为了列表方便计算时间间隔
                tmp_list = []
                for j in range(len(last_users_time)):  # 得到每一个历史物品的行为序列的时间间隔
                    tiv = float(now_user_time) / 3600.0 / 24.0 - float(last_users_time[j]) / 3600.0 / 24.0
                    tmp_list.append(str(np.sum(tiv >= gap)))
                tmp_str = ''.join(tmp_list[-3:])
                tmp_history_items_behaviors_time_list.append(tmp_str)

            # -------------------------这里需要截断，因为这里文件不截断的话训练的时候需要截断，更浪费时间--------------------------------
            history_clk_num = len(history_item_id_list)
            if (history_clk_num > 50):
                mid_id = mid_id[history_clk_num - 50:]
                cat_id = cat_id[history_clk_num - 50:]
                tiv_to_target_list = tiv_to_target_list[history_clk_num - 50:]
                tiv_to_neighbor_list = tiv_to_neighbor_list[history_clk_num - 50:]
                tmp_history_items_behaviors_list = tmp_history_items_behaviors_list[history_clk_num - 50:]
                tmp_history_items_behaviors_time_list = tmp_history_items_behaviors_time_list[history_clk_num - 50:]
            # 下面是将列表转换成字符串存入文件
            mid_str = "".join(mid_id)
            cat_str = "".join(cat_id)
            history_time_str = "".join(history_item_time_list)
            tiv_target_str = "".join(tiv_to_target_list)
            tiv_neighbor_str = "".join(tiv_to_neighbor_list)
            history_items_behaviors_str = "#".join(tmp_history_items_behaviors_list)  # 这里最多只留三个最靠近的#每个历史物品对应的购买用户A2U0LY01LK9C3EA3P0CEE0MD7MAYA33PLZ7SD5MCG0A3PD8JD9L4WEII
            history_items_behaviors_time_str = "#".join(tmp_history_items_behaviors_time_list)

            if history_clk_num >= 3:  # 8 is the average length of user behavior
                # item_id是目标物品id，item_cat是目标物品类别，mid_str，cat_str是历史行为
                if fo == ftrain:
                    flag = True  # 这里的flag是说只有在训练集中出现过的才会被选进测试集
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str + "," + history_items_behaviors_str + "," + history_items_behaviors_time_str, file=fo)
                if fo == ftest and flag:
                    print(items[1] + "," + user + "," + item_id + "," + item_cat + "," + target_time + "," + mid_str + "," + cat_str + "," + history_time_str + "," + tiv_target_str + "," + tiv_neighbor_str + "," + history_items_behaviors_str + "," + history_items_behaviors_time_str, file=fo)
        last_user = user
        # 累积历史行为
        if click:  # 只有点击了才算进历史行为
            history_item_id_list.append(item_id)
            history_item_cat_list.append(item_cat)
            history_item_time_list.append(target_time)
            history_items_behaviors_list.append(user_str)
            history_items_behaviors_time_list.append(user_time_str)


# 这里分验证集和测试集的时候，需要根据与全体数据集的比例来设定，所以先运行了一个函数，看一下训练集和测试集的数量
def print_num_of_train_and_test_before():
    train_num_line = sum([1 for l in open(f"../dataset/{file_name}/{file_name}_train_V3", "r")])
    test_num_line = sum([1 for l in open(f"../dataset/{file_name}/{file_name}_test_before_V3", "r")])
    print('train_num_line:', train_num_line)
    print('test_num_line:', test_num_line)


def part_of_test_before_to_train_and_valid():
    fi = open(f"../dataset/{file_name}/{file_name}_test_before_V3", "r")
    ftrain = open(f"../dataset/{file_name}/{file_name}_train_V3", "a")  # 注意这里是追加，不是覆盖，注意这里的执行
    ftest = open(f"../dataset/{file_name}/{file_name}_test_V3", "w")
    fvalid = open(f"../dataset/{file_name}/{file_name}_valid_V3", "w")
    # 是随机从1~10中选取整数，如果恰好是1or8，当前用户就作为验证数据集,如果恰好是2or9，当前用户就作为测试数据集
    while True:
        rand_int = random.randint(1, 10)  # 这里是前闭后闭。np.random.randint(a,b)前闭后开
        # 连续读两行
        no_clk_line = fi.readline().strip()  # 读一行
        clk_line = fi.readline().strip()  # 再读一行
        if no_clk_line == "" or clk_line == "":  # 保证正负样本成对存在
            break
        if rand_int <= 2:
            print(no_clk_line, file=ftest)
            print(clk_line, file=ftest)
        elif rand_int <= 4:
            print(no_clk_line, file=fvalid)
            print(clk_line, file=fvalid)
        else:
            print(no_clk_line, file=ftrain)
            print(clk_line, file=ftrain)


def generate_voc():
    f_train = open(f"../dataset/{file_name}/{file_name}_train_V3", "r")  # 1,A13QGAPPBC1PVS,B00E3V3CAW,Casual,1402704000,B007KR07ZYB005NKKJUSB00CQ3DJGQ,Knits & TeesArm WarmersCasual,10109
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
        tmp1 = arr[-1]
        tmp1 = tmp1.replace('#', '')
        tiv_list = arr[8] + "" + arr[9] + "" + tmp1
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

    sorted_uid_dict = sorted(uid_dict.items(), key=lambda x: x[1], reverse=True)  # 按照出现的次数排序(从多到少排序)，排序完成返回数组list
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

    pickle.dump(uid_voc, open(f"../dataset/{file_name}/uid_voc_V3.pkl", "wb"))  # 二进制文件
    pickle.dump(mid_voc, open(f"../dataset/{file_name}/mid_voc_V3.pkl", "wb"))
    pickle.dump(cat_voc, open(f"../dataset/{file_name}/cat_voc_V3.pkl", "wb"))
    pickle.dump(tiv_voc, open(f"../dataset/{file_name}/tiv_voc_V3.pkl", "wb"))
