# -*- coding: UTF-8 -*-
'''
@Project ：dien-master 
@File ：查看多少物品不在类别中.py
@Author ：小道士
@Date ：27/5/2023 下午 6:27 
'''
import pandas as pd

# 读取数据文件
df = pd.read_csv('../dataset/Electronics/item-info', header=None, names=['item_id', 'category_id'])

# 生成类别集合
given_items = set(df['item_id'])


# 读取数据文件
df = pd.read_csv('../dataset/Electronics/reviews-info', header=None, names=['user_id', 'item_id', 'timestamp'])

# 找出不在给定集合中的物品
not_in_given_items = df[~df['item_id'].isin(given_items)]['item_id'].unique()

# 打印结果
print(len(not_in_given_items))
print(not_in_given_items)
