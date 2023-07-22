import random

fi = open("../dataset/Electronics/Electronics_test_before", "r")
ftrain = open("../dataset/Electronics/local_train", "w")
ftest = open("../dataset/Electronics/local_test", "w")
# 是随机从1~10中选取整数，如果恰好是2，当前用户就作为验证数据集
while True:
    rand_int = random.randint(1, 10)
    # 连续读两行
    no_clk_line = fi.readline().strip()  # 读一行
    clk_line = fi.readline().strip()  # 再读一行
    if no_clk_line == "" or clk_line == "":  # 保证正负样本成对存在
        break
    if rand_int == 2:
        print(no_clk_line, file=ftest)
        print(clk_line, file=ftest)
    else:
        print(no_clk_line, file=ftrain)
        print(clk_line, file=ftrain)


