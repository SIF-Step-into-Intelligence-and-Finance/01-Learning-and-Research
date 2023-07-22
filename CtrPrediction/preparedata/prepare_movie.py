# -*- coding: UTF-8 -*-
'''
@Project ：dien-master 
@File ：prepare_movie.py
@Author ：小道士
@Date ：28/5/2023 下午 4:52 
'''
import pandas as pd

# 读取文件并设置列名
df = pd.read_csv('../dataset/ml_25/movies.csv', header=0, names=['movieId', 'moviename', 'cate'], sep=',')

# 保留所需列
df = df[['movieId', 'cate']]

# 将结果保存到新文件
df.to_csv('../dataset/ml-25m/meta-info', header=False, index=False, sep=',')

