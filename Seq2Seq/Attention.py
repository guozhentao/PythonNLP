"""在seq2seq模型基础上加上attention机制
本代码是在seq2seq基础上做的修改，展示了使用AttentionWrapper的方法
"""
# 下面sel.enc_cell_fw和self.enc_cell_bw定义了编码器中的前向和后向循环网络
# 它们取代了seq2seq模型代码中__init__函数里的self.enc_cell
import tensorflow as tf
self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

# 下面的代码取代了seq2seq模型代码中forward函数里的相应部分
with tf.variable_scope('encoder'):
    # 构造编码器时，使用bidirectional_dynamic_rnn来构造双向循环网络
    # 双向循环网络的顶层输出enc_outputs是一个包含两个张量的tuple，每个张量的维度都是[baich_size, max_time, HIDDEN_SIZE]
    # 代表两个LSTM在每一步的输出
    enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw, self.enc_cell_bw,
                                                             src_emb, sec_size, dtype=tf.float32)
    # 将两个LSTM的输出拼接位一个张量
    enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)   # 把要拼接的向量放在[]内，按照倒数第一个的维度进行拼接向量

with tf.variable_scope('decoder'):
    # 选择注意力机制的计算模型，使用一个隐藏层的前馈神经网络
    # memory_sequence_lenght是一个维度为[batch_size]的张量，代表batch中每个句子的长度
    # Attention需要根据这个信息把填充位置的注意力权重设置为0
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE, enc_outputs,
                                                               memory_sequence_lenght= src_size)

    # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高级的循环神经网络
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism,
                                                         attention_layer_size= HIDDEN_SIZE)
    # 使用attention_cell和dynamic_rnn构造编码器
    # 这里没有指定init_state，也就是没有使用编码器的输出来初始化输入，而是完全依赖注意力机制作为信息来源
    dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype=tf.float32)

