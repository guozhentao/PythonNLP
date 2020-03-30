"""1.引入头文件"""
"""本例的最后需要将词向量可视化出来。所以在代码行中有可视化相关的引入，即初始化，
通过设置mpl的值让plot能够显示中文信息。
Scikit-Learn的t-SNE算法模块的作用是非对称降维，是结合了t分布将高维空间的数据点映射到低维空间的距离，
主要用于可视化和理解高维数据"""
# 通过该例子练习使用nce_loss函数和word_embedding技术，实现自己的word2vec
import jieba
import numpy as np
import tensorflow as tf
import collections
from collections import Counter
import random
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
mpl.rcParams['font.family'] = 'STSong'
mpl.rcParams['font.size'] = 20

"""2.准备样本, 创建数据集"""
# 准备一段文字作为训练的样本，对其使用CBOW模型计算word2vec，并将各个词的向量关系用图展示出来

"""代码中使用get_ch_label函数将所有文字读入training_data，
然后在fenci函数里使用jieba分词库对training_data分词生成training_ci，
将training_ci放入build_dataset里并生成指定长度（350）的字典"""

training_file_path = './data/人体阴阳与电能.txt'

# 中文字
def get_ch_label(txt_file_path):   # 这个方法本质就是读取这个文件txt_file,就是返回这个文件内容
    labels = ''
    with open(txt_file_path, 'rb') as f:
        for label in f:
            # lables = lables + lable.decode('utf-8')
            labels = labels + label.decode('gb2312')    # 如果是Windows编辑的样本，编码为GB2312
    return labels       # 这实际就是读取了文件内容以返回，就是原原本本的返回文件的内容,类型是string


# 分词
def fenci(training_data):
    seg_list = jieba.cut(training_data)     # 默认是精确模式
    training_ci = ' '.join(seg_list)
    # split()：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
    training_ci = training_ci.split()   # 以空格将字符串分隔,此处的training_ci是一个列表

    training_ci = np.array(training_ci)     # 将列表转化成矩阵
    # training_ci = np.reshape(training_ci, [-1, ])    # 将矩阵training_ci转化成只有一行的列表list
    training_ci = training_ci.reshape(-1)   # 这句与training_ci = np.reshape(training_ci, [-1, ])等价
    return training_ci      # 返回的是一个list，list的元素是字符串


"""build_dataset中的实现方式是将统计词频0号位置给unknown（用UNK表示），其余按照频次由高到低排列。
unknown的获取按照预设词典大小，比如350，则频次排序靠后于350的都视为unknown"""
def build_dataset(words, n_words):    # 这个words参数就是上面返回的training_ci
    """将原始输入处理到数据集中"""
    count = [['UNK', -1]]   # 这相当于创建了一个list，这个list里面的元素也是list。没有在字典中的词编号为UNK
    # 这个count记录的是每个词对应的词频，例如[['UNK', -1] , ['人体',200] , ['电能',150],...]
    count.extend(collections.Counter(words).most_common(n_words - 1))

    dictionary = dict()   # 创建一个空字典，等同于dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)    # len(dict)是计算字典的元素总数，即键的总数
    # dictionary是一个字典：记录的是单词对应编号 即{key：单词, value：编号}(编号越小，词频越高，但第一个永远是UNK)

    data = list()
    unk_count = 0
    for word in words:      # 这个for循环就是为了得出标记为UNK的个数
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0   # dictionary['UNK']
            unk_count += 1      # 这个unk_conut就是记录遍历words时其中的word不在字典dictionary中的个数，其实就是标记为UNK的个数
        data.append(index)
    count[0][1] = unk_count

    # reversed_dictionary是一个字典：编号对应的单词  即key：编号、value：单词(编号越小，词频越高，但第一个永远是UNK)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary


training_data = get_ch_label(training_file_path)
print('总字数', len(training_data))
training_ci = fenci(training_data)
print('总词数', len(training_ci))
training_label, count, dictionary, words = build_dataset(training_ci, 350)
words_size = len(dictionary)
print('字典词数', words_size)

# print(training_label)     #将文本转为词向量
# print(words)      #每个编号对应的词
# print(dictionary)     #每个词对应的编号
# print(count)      #每个词对应的个数

print('sample data', training_label[:10], [words[i] for i in training_label[:10]])


"""3.获取批次数据
定义generate_batch函数，取一定批次的样本数据"""
data_index = 0
# generate_batch函数中使用Skip-Gram模型来构建样本，是从开始位置的一个一个字作为输入，
# 然后将其前面和后面的字作为标签，再分别组合在一起变成2组数据
# 如果是CBOW方法，根据字取标签的方法正好相反
def generate_batch(data, batch_size, num_skips, skip_window):
    # 参数说明：
    # batch_size:批次大小
    # num_skips：就是重复使用用一个单词的次数，比如 num_skips=2时，对于一句话：i love tensorflow very much。
    # 当tensorflow被选为目标词时，在产生label时要利用tensorflow两次即： tensorflow --> love， tensorflow --> very
    # skip_window：就是取上文词数，取下文词数

    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span就是窗口大小 [上文词 上文词 中心词 下文次 下文词]，通过下面的代码知道这个span为3
    buffer = collections.deque(maxlen=span)     # 就是申请了一个buffer（其实就是固定大小的窗口这里是3）
    # 即每次这个buffer队列中最多能容纳span个单词

    if data_index + span > len(data):
        data_index = 0

    buffer.extend(data[data_index:data_index + span])
    data_index += span

    # " // "来表示整数除法，返回不大于结果的一个最大的整数,而" / " 则单纯的表示浮点数除法
    for i in range(batch_size // num_skips):    # 即循环batch_size // num_skips次
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        if data_index == len(data):
            # print(data_index,len(data),span,len(data[:span]))
            # buffer[:] = data[:span]
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    # 回溯一点以避免在批处理末尾跳过单词
    data_index = (data_index + len(data) - span) % len(data)    # 防止越界
    return batch, labels    # 返回值中batch表示Skip-Gram模型的中心词，labels表示上下文中的单词，
    # 他们的形状分别为(batch_size, )和(batch_size, 1)


batch, labels = generate_batch(training_label,batch_size=8, num_skips=2, skip_window=1)

for i in range(8):      # 取第一个字，后一个是标签，再取其前一个字当标签，# 先循环8次，然后将组合好的样本与标签打印出来
    print(batch[i], words[batch[i]], '->', labels[i, 0], words[labels[i, 0]])


"""4.定义取样参数"""

"""下面代码中每批次取128个，每个词向量的维度为128，前后取词窗口为1，
num_skips表示一个input生成2个标签，nce中负采样的个数为num_sampled。
接下来是验证模型的相关参数，valid_size表示在 0- words_size/2中的数取随机不能重复的16个字来验证模型"""
batch_size = 128    # 每批次去128个
embedding_size = 128    # embedding vector的维度
skip_window = 1     # 左右取次的数量，即中心词两边各取一个
num_skips = 2   # 一个input生成两个标签

valid_size = 16
valid_window = np.int32(words_size/2)     # 取样数据的分布范围
print('valid_window', valid_window)

valid_examples = np.random.choice(valid_window, valid_size, replace=False)   # 0- words_size/2中的数取16个。不能重复
# numpy.random.choice(a, size=None, replace=True, p=None)
# 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
# replace:True表示可以取相同数字，False表示不可以取相同数字
# 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

num_sampled = 64    # 负采样的个数


"""5.定义模型变量"""
# 初始化图，为输入、标签、验证数据定义占位符，
# 定义词嵌入变量embeddings为每个字定义128维的向量，并初始化为-1～1之间的均匀分布随机数。
# tf.nn.embedding_lookup是将输入的train_inputs转成对应的128维向量embed，
# 定义nce_loss要使用的nce_weights和nce_biases
# tf.nn.nce_loss是word2vec的skip-gram模型的负例采样方式的函数
tf.reset_default_graph()

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
training_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# CPU上执行
with tf.device('/cpu:0'):
    # Look up embeddings for inputs
    embeddings = tf.Variable(tf.random_uniform([words_size, embedding_size], -1.0, 1.0))    # 94个，每个128个向量
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 计算nce的loss值
    nce_weights = tf.Variable(tf.truncated_normal([words_size, embedding_size],
                                                  stddev=1.0 / tf.sqrt(np.float32(embedding_size))))    # tf.sqrt()计算平方根
    nce_biases = tf.Variable(tf.zeros([words_size]))    # 在反向传播中，embeddings会与权重一起被nce_loss代表的loss值所优化更新。


"""6.定义损失函数和优化器"""
# 使用nce_loss计算loss，为了保证在softmax时，运算速度不受words_size过大的影响，
# 在nce中每次会产生num_sampled（64）个负样本来参与概率运算
# 优化器使用学习率为1的GradientDescentOptimizer
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=training_labels, inputs=embed,
                                     num_sampled=num_sampled, num_classes=words_size))
# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# 对embedding层做一次归一化，即正则化
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))    # keep_dims=True是不降维

normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

# 计算minibatch exemples和所有embeddings的cosine相似度
# 该相似度similarity 通过向量间夹角余弦计算
# 当cosθ为1时，表明夹角为0，即两个向量的方向完全一样。所以当cosθ的值越小，表明两个向量的方向越不一样，相似度越低
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)      # tf.matmul()实现矩阵乘法功能
print('相似度', similarity.shape)
# 验证数据取值时做了些特殊处理，将embeddings中每个词对应的向量进行平方和再开方得到norm，
# 然后将embeddings与norm相除得到normalized_embeddings。
# 当使用embedding_lookup获得自己对应normalized_embeddings中的向量valid_embeddings时，
# 将该向量与转置后的normalized_embeddings相乘得到每个词的similarity。
# 这个过程实现了一个向量间夹角余弦（Cosine）的计算

"""7.开始训练"""
num_steps = 100001      # 设置迭代次数十万零一次
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('初始化完毕')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_lables = generate_batch(training_label, batch_size, num_skips, skip_window)
        # print('batch_inputs的形状', batch_inputs.shape)
        # print('batch_lables的形状', batch_labels.shape)
        feed_dict = {train_inputs:batch_inputs, training_labels:batch_lables}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 通过打印测试可以看到  embed的值在逐渐的被调节
        # emv=sess.run(embed,feed_dict={train_inputs:[69,1494]})
        # print("emv--------------------",emv[0])
        if step % 2000 == 0:    # 每迭代2000次就输出一次loss值
            if step > 0:
                average_loss /= 2000
            print('Average_loss at the step', step, 'is :', average_loss)
            average_loss = 0

        # 将验证数据输入模型中，找出与其相近的词。这里使用了argsort函数，是将数组中的值从小到大排列后，返回每个值对应的索引
        # sim是求当前词与词典中每个词的夹角余弦，值越大则代表相似度越高

        if step % 10000 == 0:
            sim = similarity.eval(session=sess)     # 这个eval是什么意思啊？？？

            # print('Valid_size的值是：', valid_size)
            for i in range(valid_size):
                valid_word = words[valid_examples[i]]
                print('Valid_word', valid_word)
                top_k = 8  # 取排名最靠前的8个词
                nearest = (-sim[i, :]).argsort()[1:top_k+1]     # argsort函数返回的是数组值从小到大的索引值
                print('nearest', nearest, top_k)

                log_str = 'Nearest to %s:' % valid_word

                for k in range(top_k):
                    close_word = words[nearest[k]]
                    log_str = '%s, %s' % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()


"""8.词向量可视化"""
"""将词向量可视化, 在可视化之前，将词典中的词嵌入向量转成单位向量（只有方向）
然后将它们通过t-SNE降维映射到二维平面中显示"""
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


try:
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 80    # 输出100个词
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [words[i] for i in range(plot_only)]
    print('Lables:', labels)
    plot_with_labels(low_dim_embs, labels)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')