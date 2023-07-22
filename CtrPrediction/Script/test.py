# -*- coding: UTF-8 -*-
'''
@Project ：Mymodel 
@File ：test.py
@Author ：小道士
@Date ：18/6/2023 下午 7:58 
'''
import tensorflow as tf

# 假设我们有一个形状为(10, 128)的target_item_emb
target_item_emb = tf.random.normal(shape=(10, 128))

print('Before expand_dims, shape:', target_item_emb.shape)

# 在第一维（索引从0开始）上增加一个新的维度
target_item_emb_expanded = tf.expand_dims(target_item_emb, 1)

print('After expand_dims, shape:', target_item_emb_expanded.shape)
