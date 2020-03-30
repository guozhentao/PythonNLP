import tensorflow as tf

max_len = 50    # 限定句子的最大单词数量
sos_id = 1      # 目标语言词汇表中<sos>的ID，即行索引

# 使用Dataset从一个文件中读取一个语言的数据
# 数据的格式为每行一句话，单词已经转化为单词编号。
# Dataset可以看作是相同类型“元素”的有序列表。在实际使用时，单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或者dict
def make_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)

    # 根据空格将单词编号切分开，并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # 将字符串形式的单词编号转化成整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))

    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中
    dataset = dataset.map(lambda x: (x, tf.size(x)))

    return dataset      # 每个句子－对应的长度组成的TextLineDataset类的数据集对应的张量


# src_train_data = './data/train_word_number_en.txt'    # 源语言输入文件
# trg_train_data = './data/train_word_number_zh.txt'    # 目标语言输入文件
# src_data = make_dataset(src_train_data)
# print('src_data:', src_data)
# trg_data = make_dataset(trg_train_data)
# print('trg_data:', trg_data)

# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和batching操作
def make_src_trg_dataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据
    src_data = make_dataset(src_path)
    trg_data = make_dataset(trg_path)

    # 通过zip操作将两个Dataset合并为一个Dataset。
    # 现在每个Dataset中每一项数据ds都有4个张量组成，
    # ds[0][0]是源句子
    # ds[0][1]是源句子长度
    # ds[1][0]是目标句子
    # ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))     # 使用zip()函数时候，注意要把多个dataset用括号包起来

    # 删除内容为空(只包含<eos>)的句子和长度过长的句子
    def filter_length(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)

        # tf.logical_and() 相当于与操作，后面两个都为true最终结果才会为true，否则为false
        # tf.greater(src_len, 1)指句子长度必须得大于1也就是不能为空的句子
        # tf.less_equal(src_len, max_len)指长度要小于最大长度
        src_len_ok = tf.logical_and(tf.greater(src_len, 1),               # src_len大于1，返回True
                                    tf.less_equal(src_len, max_len)       # trg_len小于MAX_LEN，返回True
                                    )
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, max_len))

        return tf.logical_and(src_len_ok, trg_len_ok)

    # filter接收一个函数并将该函数作用于dataset的每个元素，根据返回值True或False保留或丢弃该元素，
    # True保留该元素，False丢弃该元素
    # 最后得到的就是去掉空句子和过长的句子的数据集
    dataset = dataset.filter(filter_length)

    # 解码器需要两种格式的目标句子：
    # 1.解码器的输入(trg_input), 形式如同'<sos> X Y Z'
    # 2.解码器的目标输出(trg_label), 形式如同'X Y Z <eos>'
    # 上面从文件中读到的目标句子是'X Y Z <eos>'的形式，我们需要从中生成'<sos> X Y Z'形式并加入到Dataset
    # 编码器只有输入,没有输出,而解码器有输入也有输出，输入为<sos>＋(除去最后一位eos的label列表)
    # 例如train_word_number_en.txt中每行的最后都是2，id为２的就是<eos>
    def make_trg_input(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[sos_id], trg_label[:-1]], axis=0)   # tf.concat()就是把几个张量按照axis拼接起来
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(make_trg_input)

    # 随机打乱训练数据
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度
    padded_shapes = ((tf.TensorShape([None]),   # 源句子是长度未知的向量
                      tf.TensorShape([])),      # 源句子长度是单个数字
                     (tf.TensorShape([None]),   # 目标句子（解码器输入）是长度未知的向量
                      tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
                     tf.TensorShape([]))         # 目标句子长度是单个数字
                     )
    # 调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset