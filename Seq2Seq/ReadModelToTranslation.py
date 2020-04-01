import tensorflow as tf
import codecs
"""之前的程序已经完成了机器翻译模型的训练步骤，并将训练好的模型保存在checkpoint中，
下面的程序是从checkpoint中读取模型并对一个新句子进行翻译。
对新句子进行翻译的过程也被称之为解码或推理"""
"""在解码的过程中，解码器的实现与训练时有很大的不同。这是因为训练时解码器可以从输入中读取完整的目标句子，
因此可以使用dynamic_rnn简单的展开成前馈网络，而在解码过程中，模型只看到输入句子，却看不到目标句子。
解码器在第一步读取<sos>符号后，预测目标句子的第一个单词，然后需要将这个预测的单词复制到第二步作为输入，再预测第二个单词，
直到预测的单词为<eos>为止。
这个过程需要一个循环结构来实现，在TensorFlow中这种循环结构由tf.while_loop来实现。"""
# 下面简单说一下tf.while_loop()函数
# final_state = tf.while_loop(cond, loop_body, init_state)
# cond是一个函数，负责判断继续执行循环的条件
# loop_body是每个循环提内执行的操作，负责对循环状态进行更新。
# init_state为循环的起始状态，它可以包含多个Tensor或者TensorArray
# 返回的结果是循环结束时的循环状态

"""一下程序为解码过程"""
# 读取checkpoint的路径。9000表示训练程序在第9000步保存的checkpoint
CHECKPOINT_PATH = './checkpoint_model/seq2seq_ckpt-200'

# 模型参数必须与训练时的模型参数保持一致
HIDDEN_SIZE = 1024      # LSTM隐藏层的规模
NUM_LAYERS = 2      # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 10000      # 源语言词汇表大小
TRG_VOCAB_SIZE = 10000      # 目标语言词汇表大小
SHARE_EMB_AND_SOFTMAX = True    # 在softmax层和词向量层之间共享参数

# 词汇表中<sos>和<eos>的ID。在解码过程中需要<sos>作为第一步的输入，并检查是否是<eos>，因此需要知道这两个符号的ID
SOS_ID = 1
EOS_ID = 2

# 定义NMT model类来描述模型
class NMTModel(object):
    # 与训练时的__init__()函数相同。通常在训练程序和解码程序过程中复用NMTMedel类及其__init__函数，
    # 这样可以确保解码时和训练时定义的变量相同。
    def __init__(self):
        # 定义编码器和解码器所使用的的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable('nmt_model/src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('nmt_model/trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)   # tf.transpose()对张量进行转置操作
        else:
            self.softmax_weight = tf.get_variable('nmt_model/weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('nmt_model/softmax_bias', [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        # 虽然解码器输入的只有一个句子，但是因为dynamic_rnn要求输入的是batch的形式，因此这里需要将输入句子
        # 整理为大小为1的batch
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        # 使用dynamic_rnn构造编码器。这一步与训练时的相同
        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell, inputs=src_emb,
                                                       sequence_length=src_size, dtype=tf.float32)
        # 设置解码器的最大步数。这是为了避免在极端情况下出现无限循环的问题
        MAX_DEC_LEN = 100

        with tf.variable_scope('decoder/rnn/multi_rnn_cell'):

            # 使用一个变长的TensorArray来存储生成的句子
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True,clear_after_read=False)

            # 填入第一个单词<sos>作为解码器的输入
            init_array = init_array.write(0, SOS_ID)

            # 构建初始的循环状态。
            # 循环状态包括循环神经网络的隐藏状态，保存生成句子的TensorArray,以及记录解码步数的一个整数step
            init_loop_var = (enc_state, init_array, 0)

            # tf.while_loop的循环条件
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(
                                      tf.logical_and(
                                           tf.not_equal(trg_ids.read(step), EOS_ID),
                                           tf.less(step, MAX_DEC_LEN-1)
                                                    )
                                     )

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

                # 这里不使用dynamic_rnn，而是直接调用dec_cell向前计算一步
                dec_outputs, next_state = self.dec_cell.call(state=state, inputs=trg_emb)

                # 计算每个可能的输出单词对应的logit，并选取logit最大的单词作为这一步的输出
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)

                # 将这一步的、输出的单词写入循环状态的trg_ids中。
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step+1

        # 执行tf.while_loop,返回最终状态
        state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)
        return trg_ids.stack()

def main():
    # 定义训练用到的循环神经网络模型
    # with tf.variable_scope('nmt_model', reuse=None):
    model = NMTModel()

    # 定义一个测试例子。这里的测试例子是经过预处理后的“This is a test.”
    test_sentence = [90, 13, 9, 689, 4, 2]
    # test_sentence = [11, 24, 9, 182, 183, 4, 2]     # I have a great idea.

    # 建立解码器所用到的计算图
    output_op = model.inference(test_sentence)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)    # 恢复模型及其所有变量
    # 读取翻译结果
    output = sess.run(output_op)
    print(output)
    sess.close()


if __name__ == '__main__':
    main()
