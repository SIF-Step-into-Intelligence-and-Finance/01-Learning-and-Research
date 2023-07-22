# -*- coding: UTF-8 -*-
'''
@Project ：Mymodel 
@File ：prepare_amazon_new.py
@Author ：小道士
@Date ：11/6/2023 上午 9:28 
'''
import pandas as pd

#因为前面的数据比较稀疏，所以选择2014年之后的数据
def choose_reviews_after_2014():
    # 读取原始文件
    df = pd.read_csv('../dataset/Electronics_new/reviews-info-all', header=None, names=['user_id', 'item_id', 'timestamp'])
    # 将时间戳列转换为日期格式
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

    # 筛选购买时间在2014年1月1日之后的记录
    filtered_df = df[df['date'] >= pd.Timestamp('2015-01-01')]

    # 将筛选结果保存到新的文件
    filtered_df[['user_id', 'item_id', 'timestamp']].to_csv('../dataset/Electronics_new/reviews-info', index=False, header=False)
choose_reviews_after_2014()
num_line = sum([1 for _ in open("../dataset/Electronics_new/reviews-info", "r")])
print(num_line)

df = pd.read_csv('../dataset/Electronics_new/reviews-info', header=None, names=['user_id', 'item_id', 'timestamp'])
unique_values1 = df['user_id'].nunique()
unique_values2 = df['item_id'].nunique()
print(unique_values1,unique_values2)