import numpy as np
import tensorflow as tf

train_data_path = './data/ptb_train_id.txt'     # 使用单词编号表示的训练数据
train_batch_size = 20
train_num_step = 35

# 从文件中读取数据，并返回包含单词编号的数组
def read_data(file_path):
    with open(file_path, 'r') as fin:
        # 将整个文档读进一个长字符串
        id_tring = ' '.join([line.strip() for line in fin.readlines()])
        # print('id_string是', id_tring)
    id_list = [int(w) for w in id_tring.split()]    # 将读取的单词编号转为整数，# id_list的长度是929589
    # print('id_list是', id_list)
    return id_list


def make_batchs(id_list, batch_size, num_step):     # batch_size=20, num_step=35
    # 计算总的batch数量，每个batch包含的单词数是batch_size*num_step
    num_batches = (len(id_list)) // (batch_size*num_step)   # num_batches是 1327

    # 将数据整理成一个维度为[batch_size, num_batches*num_step]的二维数组
    data = np.array(id_list[: num_batches*batch_size*num_step])
    data = np.reshape(data, [batch_size, num_batches*num_step])

    # 沿着第二个维度将数据切分成num_batch个batch，存入一个数组
    # axis=1就是竖着连接或者竖着劈开,这个地方是按照列劈开成了num_batches份
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位， 这里得到的是RNN每一步输出所需要预测的下一个单词
    label = np.array(id_list[1: num_batches*batch_size*num_step+1])
    label = np.reshape(label, [batch_size, num_batches*num_step])
    label_batches = np.split(label, num_batches, axis=1)

    # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵
    return list(zip(data_batches, label_batches))


id_list = read_data(train_data_path)
li = make_batchs(id_list, 20, 35)



