from Seq2SeqDatasetBatchingMethod import make_src_trg_dataset
import tensorflow as tf
"""双层LSTM 作为循环神经网络的主体,并在 Softmax 层和词向量层之间共享参数，：
增加了一个循环神经网络作为编码器，
使用 Dataset 动态读取数据，而不是直接将所有数据读入内存（这个就是Dataset输入数据的特点）
每个 batch 完全独立，不需要在batch之间传递状态（因为不是一个文件整条句子，每个句子之间没有传递关系）
每训练200步便将模型参数保存到一个 checkpoint 中，以后用于测试。"""

# 直接使用已经转化为单词编号的文件
SRC_TRAIN_DATA = './data/train_word_number_en.txt'  # 源语言输入文件
TRG_TRAIN_DATA = './data/train_word_number_zh.txt'  # 目标语言输入文件
CHECKPOINT_PATH = './checkpoint_model/seq2seq_ckpt'      # checkpoint的保存路径，训练好的模型放在checkpoint中，checkpoint：n.检查站
HIDDEN_SIZE = 1024      # LSTM隐藏层的大小
NUM_LAYERS = 2      # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 10000      # 源语言词汇表大小
TRG_VOCAB_SIZE = 10000       # 目标语言词汇表大小
BATCH_SIZE = 100        # 训练数据batch的大小
NUM_EPOCH = 5       # 使用训练集的轮数
KEEP_PROB = 0.8     # 节点不被dropout的概率
MAX_GRAD_NORM = 5   # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True      # 在softmax层和词向量层之间共享参数


# 使用神经网络机器翻译模型来描述模型：
# 定义NMT model类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器和解码器所使用的的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable('src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)   # tf.transpose()对张量进行转置操作
        else:
            self.softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('softmax_bias', [TRG_VOCAB_SIZE])

    # 在forward函数中定义模型的前向计算图
    # src_input, src_size, trg_input, trg_lable, trg_size
    # 是Seq2SeqDatasetBatchingMethod文件make_src_trg_dataset方法产生的5种张量
    def forward(self, src_input, src_size, trg_input, trg_lable, trg_size):

        # 对tf.shape()函数的理解：
        # tf.shape(input,name=None,out_type=tf.int32)把input张量转化成类型是tf.int32类型的张量，默认是tf.int32类型。
        # 如果在运行下面代码的时候，
        # x = tf.placeholder(tf.float32, shape=[None, 227] )
        # 想知道None到底是多少，这时候，只能通过tf.shape(x)[0]这种方式来获得。
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出编号转化为词向量
        # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 使用dynamic_rnn构造编码器
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state。
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类的tuple,每个LSTMStateTuple对应编码器中一层的状态
        # enc_outputs是顶层LSTM在每一步的输出，他的维度是[batch_size, max_time, HIDDEN_SIZE], seq2seq模型中不会用到enc_outputs,
        # 而后面介绍的attention模型会用到它。
        with tf.variable_scope('encoder'):      # 用于定义创建变量（层）的操作的上下文管理器。
            enc_outpus, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell, inputs=src_emb,
                                                      sequence_length=src_size, dtype=tf.float32)

        # 使用dynamic_rnn构造解码器
        # 解码器读取目标句子的每个位置的词向量，输出的dec_outputs为每一步顶层LSTM的输出。
        # dec_outputs的维度是[batch_size, max_time, HIDDEN_SIZE]。
        # initial_state=enc_state表示用编码器的输出来初始化第一步的隐藏状态。
        with tf.variable_scope('decoder'):
            dec_outputs, _ = tf.nn.dynamic_rnn(cell=self.dec_cell, inputs=trg_emb,
                                               sequence_length=trg_size, initial_state=enc_state)
        # 计算解码器每一步的log perplexity。
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])     # 输出重新转换成shape为[,HIDDEN_SIZE]
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias     # 计算解码器每一步的softmax概率值
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_lable, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练
        label_weights = tf.sequence_mask(lengths=trg_size, maxlen=tf.shape(trg_lable)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss*label_weights)    # tf.reduce_sum()对里面的参数的元素求和
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作
        trainable_variable = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练模型
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variable)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variable))

        return cost_per_token, train_op

# 使用给定的模型model上训练一个epoch，并返回全局步数
# 每训练200步就保存一个checkpoint。
def run_epoch(session, cost_op, train_op, saver, step):
    # 训练一个epoch
    # 重复训练步骤，直至遍历完Dataset中的所有数据
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供
            cost, _ = session.run([cost_op, train_op])
            if step % 20 == 0:
                print('在%d步后，每个token的cost是%.3f' % (step, cost))

            # 每500步保存一个checkpoint
            if step % 100 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)

            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练用的循环神经网络模型
    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel()
    # 定义输入数据
    data = make_src_trg_dataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图，输入数据以张量的形式提供给forward函数
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    # 训练模型
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print('迭代中：%d' % (i+1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__=='__main__':
    main()




