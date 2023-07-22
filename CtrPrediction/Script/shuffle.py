import random

def shuffle_file(input_file, output_file_path):
    # 读取输入文件的所有行
    with open(input_file, 'r') as file_in:
        lines = file_in.readlines()
    # # 打乱行的顺序
    random.shuffle(lines)
    # # 将打乱后的行写入输出文件
    with open(output_file_path, 'w') as file_out:
        file_out.writelines(lines)
    return output_file_path

