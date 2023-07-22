# -*- coding: UTF-8 -*-
'''
@Project ：dien-master 
@File ：split_test_by_time.py
@Author ：小道士
@Date ：23/4/2023 下午 10:21 
'''
# file_input_valid=open("../dataset/books/books_valid_V1","r")
# file_input_test=open("../dataset/books/books_test_V1","r")
# file_output=open("../dataset/books/books_valid_V1_part","w")
# for line in file_input_valid:
#     items=line.strip().split(',')
#     if items[4]>="1397116033":
#         print(line.strip(),file=file_output)
# for line in file_input_valid:
#     items=line.strip().split(',')
#     if items[4]>="1397116033":
#         print(line.strip(),file=file_output)

num_line = sum([1 for l in open("../dataset/books/books_valid_V1_part", "r")])
print(num_line)
