import json
import math
import os
import pickle as pkl
import random

import shuffle


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return json.load(f)  # <class 'dict'>
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


class DataIterator:  # 迭代器的定义：类中必须定义__init__和__next__方法，并且__iter__方法返回对象本身，__next__方法返回下一个数据，如果没有数据了，需要抛出StopIteration异常
    counter = 0  # 类变量，用于存储计数器的值
    def __init__(self, source, uid_voc, mid_voc, cat_voc, tiv_voc, batch_size=128, maxlen=50, skip_empty=True, max_batch_len=1000, minlen=None, shuffle_each_epoch=False,args=None):
        self.source_dicts = []
        for source_dict in [uid_voc, mid_voc, cat_voc, tiv_voc]:
            self.source_dicts.append(load_dict(source_dict))
        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])
        self.n_tiv = len(self.source_dicts[3])
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty
        self.source_file = source  # 记录文件的名字
        self.source_buffer = []
        self.buffer_size = batch_size * max_batch_len
        self.end_of_data = False
        self.shuffle = shuffle_each_epoch
        self.model_name= args.train_model
        if shuffle_each_epoch:
            self.increment_counter()
            self.source_orig_path = source
            self.tmp_file_path = shuffle.shuffle_file(self.source_orig_path, self.source_orig_path + '_shuffle' +self.model_name)
            self.source = open(self.tmp_file_path, 'r')
        else:
            self.source = open(source, 'r')
        # -------------------------------构建 item id 和 category id 之间的映射关系，之前都是string类型的映射，现在改成index映射-------------------------------
        file_meta = open(os.path.dirname(source) + "/item-info", "r", encoding='utf-8')
        meta_map = {}  # 存放的是物品以及对应的类别
        self.meta_id_map = {}  # 存放的是物品以及对应的类别，只不过这个时候要换成id数值类型
        for line in file_meta:
            arr = line.strip().split(",")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
                if arr[0] in self.source_dicts[1]:
                    mid_idx = self.source_dicts[1][arr[0]]
                else:
                    mid_idx = 0
                if arr[1] in self.source_dicts[2]:
                    cat_idx = self.source_dicts[2][arr[1]]
                else:
                    cat_idx = 0  # 如果不存在就设为默认类别
                self.meta_id_map[mid_idx] = cat_idx
        # -------------------------------------从 "reviews-info" 读取，根据出现的频次生成辅助损失所需要的负样本所需要的id list-------------------------------------
        file_reviews = open(os.path.dirname(source) + "/reviews-info", "r", encoding='utf-8')
        self.itemid_list_for_random = []
        for line in file_reviews:
            arr = line.strip().split(",")
            item_idx = 0
            if arr[1] in self.source_dicts[1]:  # 如果在这里面就变为id，否则就是上面默认的0
                item_idx = self.source_dicts[1][arr[1]]  # 得到物品id
            self.itemid_list_for_random.append(item_idx)

    @classmethod
    def increment_counter(cls):
        cls.counter += 1

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat, self.n_tiv

    def reset(self):
        if self.shuffle:
            self.tmp_file_path = shuffle.shuffle_file(self.source_orig_path, self.source_orig_path + '_shuffle' + self.model_name)
            self.source = open(self.tmp_file_path, 'r')
        else:
            self.source.seek(0)

    def __iter__(self):  # 返回对象本身
        return self

    def __next__(self):
        if self.end_of_data:
            # 读完一次文件，重新来过
            self.reset()
            self.end_of_data = False
            raise StopIteration
        # 每次batch需要返回的列表
        source = []
        target = []
        # 如果 self.source_buffer没有数据，则从样本文件读取k行。可以理解为一次性读取最大buffer。
        if len(self.source_buffer) == 0:
            for k_ in range(self.buffer_size):
                line = self.source.readline()  # 这里文件会记录读取的位置，会接着读取，直到读到末尾“”空字符
                if line == "":  # 读取到文件末尾
                    break
                self.source_buffer.append(line.strip("\n").split(","))  # [1,A1RLQXYNCMWRWN,B004DFG0LG,Top-Handle Bags,1330992000,B004I72CMUB004QVDFWOB0007YVP1W,Knits & TeesBlouses & Button-Down ShirtsJeans,132744960013274496001330992000,10100,0100]
        if len(self.source_buffer) == 0:
            self.reset()
            self.end_of_data = False
            raise StopIteration
        try:
            while True:  # 等到达到128就跳出去
                try:  # pop是从后边开始读的
                    items = self.source_buffer.pop()  # 1,A1RLQXYNCMWRWN,B004DFG0LG,Top-Handle Bags,1330992000,B004I72CMUB004QVDFWOB0007YVP1W,Knits & TeesBlouses & Button-Down ShirtsJeans,132744960013274496001330992000,10100,0100
                except IndexError:  # 读到了最后，不到128也会跳出去
                    break
                uid = self.source_dicts[0][items[1]] if items[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][items[2]] if items[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][items[3]] if items[3] in self.source_dicts[2] else 0
                # 4是目标物品时间戳
                target_time = items[4]
                mid_list = [self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0 for fea in items[5].split("")]  # 历史行为物品B007KR07ZYB005NKKJUSB00CQ3DJGQ
                if self.skip_empty and (not mid_list):  # 如果用户没有历史行为就过滤掉
                    continue
                if self.minlen is not None and len(mid_list) <= self.minlen:  # 如果一个用户的行为序列过小就过滤掉
                    continue
                if self.maxlen is not None and len(mid_list) > self.maxlen:  # 如果一个用户的行为序列过大就过滤掉
                    continue
                cat_list = [self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0 for fea in items[6].split("")]  # 历史行为物品类别Knits & TeesArm WarmersCasual
                history_time_list = [fea for fea in items[7].split("")]  # 历史行为物品的时间
                tiv_target_list = [self.source_dicts[3][fea] for fea in items[8].split("")]  # 历史行为物品距离目标物品时间间隔10109
                tiv_neighbor_list = [self.source_dicts[3][fea] for fea in items[9].split("")]  # 历史行为物品相邻之间的时间间隔0013
                item_behaviors_list = []  # 每个历史物品对应的行为序列
                for fea1 in items[10].split("#"):  # 每个历史物品
                    tmp_user_list = [self.source_dicts[0][fea2] if fea2 in self.source_dicts[0] else 0 for fea2 in fea1.split("")]
                    tmp_user_list.extend([0] * (3 - len(tmp_user_list)))  # 这里需要填充一下，物品行为序列不足3个用户的
                    item_behaviors_list.append(tmp_user_list)  # 列表的列表；第一维对应每个历史物品，第二维对应每个历史物品的购买用户

                item_behaviors_time_list = []  # 每个历史物品对应的行为序列的时间
                for fea1 in items[11].split("#"):  # 每个历史物品（对应多个用户）
                    tmp_time_list = [self.source_dicts[3][fea2] for fea2 in fea1.split("")]  # 每个用户的行为
                    tmp_time_list.extend([18] * (3 - len(tmp_time_list)))  # 18是时间间隔的最大值
                    item_behaviors_time_list.append(tmp_time_list)  # 列表的列表；第一维对应每个历史物品，第二维对应每个历史物品的购买时间

                history_user_buy_items_list = []
                for fea1 in items[12].split("#"):  # 每个历史物品
                    tmp_history_user_buy_items_list = []
                    for fea2 in fea1.split(""):  # 每个用户
                        tmp_list = [self.source_dicts[1][fea3] for fea3 in fea2.split("@")]
                        tmp_list.extend([0] * (3 - len(tmp_list)))  # 用户没有买够3个物品，这里填充成默认的
                        tmp_history_user_buy_items_list.append(tmp_list)
                    tmp_history_user_buy_items_list.extend([[0, 0, 0]] * (3 - len(tmp_history_user_buy_items_list)))  # 物品没有被三个用户交互
                    history_user_buy_items_list.append(tmp_history_user_buy_items_list)

                history_user_buy_cats_list = []
                for fea1 in items[13].split("#"):  # 每个历史物品
                    tmp_history_user_buy_cats_list = []
                    for fea2 in fea1.split(""):  # 每个用户
                        tmp_list = [self.source_dicts[2][fea3] for fea3 in fea2.split("@")]
                        tmp_list.extend([0] * (3 - len(tmp_list)))  # 用户没有买够3个物品，这里填充一下
                        tmp_history_user_buy_cats_list.append(tmp_list)
                    tmp_history_user_buy_cats_list.extend([[0, 0, 0]] * (3 - len(tmp_history_user_buy_cats_list)))  # 物品没有被三个用户交互
                    history_user_buy_cats_list.append(tmp_history_user_buy_cats_list)

                # DIEN的辅助损失需要采样负样本
                # 针对mid_list中的每一个pos_mid，制造5个负采样历史行为数据；具体就是从 itemid_list_for_random 中随机获取5个id（如果与pos_mid相同则再次获取新的）；
                # 即对于用户的每一个历史行为，代码中选取了5个样本作为负样本
                noclk_mid_list = []
                noclk_cat_list = []
                for pos_mid in mid_list:  # 遍历每一个历史行为，对每一个历史行为都要采样相应的负样本
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(0, len(self.itemid_list_for_random) - 1)
                        noclk_mid = self.itemid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)  # 列表的列表：每一个正行为都有一列表（5个）的负行为
                    noclk_cat_list.append(noclk_tmp_cat)

                source.append([uid, mid, cat, target_time, mid_list, cat_list, history_time_list, tiv_target_list, tiv_neighbor_list, item_behaviors_list, item_behaviors_time_list, noclk_mid_list, noclk_cat_list, history_user_buy_items_list, history_user_buy_cats_list])
                target.append([float(items[0]), 1 - float(items[0])])
                # 如果够一个batch就返回
                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) == 0 or len(target) == 0:
            source, target = self.__next__()

        # 这里抛弃最后一个不足batch_size的样本
        if len(source) < self.batch_size or len(target) < self.batch_size:
            source, target = self.__next__()
        # 每一个batch打乱
        combined = list(zip(source, target))
        random.shuffle(combined)
        source, target = zip(*combined)
        return source, target  # [uid, mid, cat, target_time, mid_list, cat_list, history_time_list, tiv_target_list, tiv_neighbor_list, item_behaviors_list,item_behaviors_time_list,noclk_mid_list, noclk_cat_list]

    def len(self):
        return math.ceil(sum([1 for _ in open(self.source_file, 'r')]) / self.batch_size)
