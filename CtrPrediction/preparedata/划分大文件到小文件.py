# -*- coding: UTF-8 -*-
'''
@Project ：dien-master 
@File ：划分大文件到小文件.py
@Author ：小道士
@Date ：31/5/2023 下午 1:42 
'''

import os


# 定义文件划分的函数
def split_file_by_user(file_path, output_directory, num_files):
    # 创建输出目录（如果不存在）
    os.makedirs(output_directory, exist_ok=True)

    # 创建字典来存储每个用户的数据
    user_data = {}

    # 逐行读取原始文件
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split(',')
            user_id = columns[2]  # 第三列是用户ID
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(line)
    # 计算每个文件应包含的用户数量
    num_users = len(user_data)
    users_per_file = num_users // num_files
    # 划分用户数据并写入文件
    file_count = 1
    user_count = 0
    for user_id, data in user_data.items():
        if user_count == 0:
            # 创建新的输出文件
            file_name = f'output_{file_count}.txt'
            file_path = os.path.join(output_directory, file_name)
            output_file = open(file_path, 'w')
        # 写入用户数据到当前文件
        for line in data:
            output_file.write(line)
        user_count += 1
        if user_count == users_per_file:
            # 关闭当前文件
            output_file.close()
            # 重置计数器
            user_count = 0
            file_count += 1
    # 关闭最后一个文件（如果有剩余的用户）
    if not output_file.closed:
        output_file.close()

# 调用函数进行文件划分
split_file_by_user('../dataset/books/books_last_samples', './books_last_samples_split', 10)
