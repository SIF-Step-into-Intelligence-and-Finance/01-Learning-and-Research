# -*- coding: UTF-8 -*-
'''
@Project ：dien-master 
@File ：生成物品行为序列.py
@Author ：小道士
@Date ：21/5/2023 上午 10:25 
'''
import pandas as pd

# 读取文件
df = pd.read_csv('../dataset/Clothing/reviews-info', header=None)

# 重命名列名
df.columns = ['user_id', 'item_id', 'timestamp']

# 按物品id分组，并按照购买时间排序，生成新的DataFrame
new_df = df.groupby('item_id').apply(lambda x: x.sort_values('timestamp')).reset_index(drop=True)

# 生成新的文件
new_df.to_csv('output_file_Clothing.csv', header=False, index=False, columns=['item_id', 'user_id', 'timestamp'])