import numpy as np
def get_cut_timestamp(train_percent=0.85,file_path=None):
  time_list = []
  file_input = open(file_path, "r")
  samples_count = sum([1 for l in open(file_path, "r")])#获取sample的长度
  train_size = int(samples_count * train_percent)
  for line in file_input:
    time = float(line.strip().split("\t")[-1])
    time_list.append(time)
  index = np.argsort(time_list, axis=-1)#返回的是数组值从小到大排序的索引值
  cut_time_index = index[train_size]
  return time_list[cut_time_index]