import tensorflow as tf
import numpy as np
"""word2vec将词转化为词向量
   以下代码是源码
   博客网址https://blog.csdn.net/qq_41076797/article/details/99690725
   """

# 1.定义一个函数用来下载数据集
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
    # 如果文件不存在就下载
    import os
    if not os.path.exists(filename):
        from six.moves import urllib    # six.moves模块就是为了兼容python2.x和3.x版本的，不然会因为下载网址而出错
        filename, _ = urllib.request.urlretrieve(url+filename, filename)
        print('文件名是：', filename)
    # 获取文件相关属性
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('已下载改文件：', filename)
    else:
        print('文件的格式大小为：', statinfo.st_size)
        raise Exception(
            filename + '验证失败，您可以使用浏览器访问它吗？谢谢！'
        )
    return filename


filename = maybe_download('text8.zip', 31344016)      # 调用方法下载数据集，获得数据集

# 2.下载了数据后，读取该数据
def read_data(filename):
    import zipfile
    with zipfile.ZipFile(filename) as f:    # zipfile.ZipFile()就是解压缩文件
        import tensorflow as tf
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        # tf.compat.as_str()实现python2和python3对字符串处理的兼容性，将bytes或者Unicode的字符串都转换为unicode字符串
    return data


words_data = read_data(filename)     # 有了数据集后，调用方法读取数据
print('words_data内容是：', words_data)
print('words_data的类型是：', type(words_data))
print('数据集大小是：', len(words_data))


# 3.对数据进行处理，建立字典，并用UNK token代替稀有词(即词频非常小的词)
# 按照词频只留前50000个单词，其他低频词的都归为UNK
vocabulary_size = 50000
def build_dataset(words_data, vocabulary_size):
    count = [['UNK', -1]]   # 这个count是一个list，记录的是每个词对应的词频,这个list的元素仍然是一个list
                            # 例如[['UNK', -1] , ['the',1061396] , ['of',593677],['and', 416629],...]可以把count理解为字典
    # 列表里已经有一个UNK了，所以再统计49999个
    import collections
    count.extend(collections.Counter(words_data).most_common(vocabulary_size-1))      # 这个count里的元素按照词频已经排好序了
    # return count
# count1 = build_dataset(words, 50000)
# print(count1)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # 把words_data里面的单词进行编号，单词所对应的的编号是其词频
    data = list()
    unk_count = 0
    for word in words_data:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)    # 这个data中存放的是语料文件中单词对应的编号
    print('遍历完语料文件后，被标记为unk的个数为：', unk_count)
    count[0][1] = unk_count     # 记录整个语料文件中被标记为unk的个数

    # 把dictionary翻转一下
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个元组，然后返回由这些元组组成的列表
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words_data=words_data, vocabulary_size=50000)


"""3.生成skip-gram模型的批次数据"""
data_index = 0
# batch_size 批次大小
# num_skips是生成多少次label，其实就是中心词用几次
# skip_window：窗口大小等于2*skip_window+1
def generate_batch(batch_size, num_skips, skip_window):     # 下面会调用这个函数，对应的参数为8,2,1
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window

    import numpy as np
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2*skip_window + 1    # 这个span就是窗口大小，[上文词 中心词 下文词]，通过下面的代码知道这个span为3

    # 定义一个双向队列
    import collections
    buffer = collections.deque(maxlen=span)    # 固定了长度为3，满员后进队会使得队首自动出队

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 获取batch和labels
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]

        for j in range(num_skips):    # 循环两次，目标词对应一个上文词一个下文词
            while target in targets_to_avoid:
                import random
                target = random.randint(0, span-1)     # random.randint(a,b)用于生成一个随机整数x（a<=x<=b）
                                                       # np.random.randint(a,b)也是用于生成一个随机整数，范围是a<=x<b
            targets_to_avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_window]
            labels[i*num_skips+j, 0] = buffer[target]
        buffer.append(data[data_index])    # 队首出队，向后移动一位
        data_index = (data_index+1) % len(data)
    data_index = (data_index+len(data)-span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):  # 打印一个批次（即batch_size个）的样本
    print(batch[i], reverse_dictionary[batch[i]], '-->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# 定义取样参数
batch_size = 128
embedding_size = 128    # 词向量的维度
num_skips = 2
skip_window = 1

valid_size = 16    # 用于评估相似性的随机单词集大小
valid_window = 100

valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# numpy.random.choice(a, size=None, replace=True, p=None)
# 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
# replace:True表示可以取相同数字，False表示不可以取相同数字
# 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

num_sample = 64    # 负采样的个数

"""4.建立并且训练skip-gram模型"""

graph = tf.Graph()    # tf.Graph()它可以通过tensorboard用图形化界面展示出来流程结构.
                      # 2. 它可以整合一段代码为一个整体存在于一个图中
                      # 表示实例化了一个类
# tf.Graph()表示实例化了一个类，一个用于 tensorflow 计算和表示用的数据流图，
# 通俗来讲就是：在代码中添加的操作（画中的结点）和数据（画中的线条）都是画在纸上的“画”，
# 而图就是呈现这些画的纸，你可以利用很多线程生成很多张图，但是默认图就只有一张。

with graph.as_default():    # tf.Graph().as_default()表示将这个类实例，也就是新生成的图作为整个tensorflow运行环境的默认图
    # 输入数据
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])       # 一个批次的数据的编号
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])    # [128,1]
    # 验证集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 在CPU上执行
    with tf.device('/cpu:0'):
        # embeddings每次词对应的词向量矩阵
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))   # 50000*128的矩阵，值在-1到1之间均匀分布
        # tf.nn.embedding_lookup(params,ids)其实就是按照ids顺序返回params中的第ids行
        # 比如说，ids=[1,7,4]就是返回params中第1，7，4行，返回结果为params的1，7，4行组成的tensor
        # 提取要训练的词  并不是五万个词都训练一起  下面就是从所有词中抽取我们要训练的
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)   # ([50000,128],[128])返回一个[128,128]的矩阵
        print('我就是想看看这个train_inputs是什么：', train_inputs)

        import math
        nce_weight = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        # tf.truncated_normal()从截断的正态分布中输出随机值，组成一个矩阵，形状是[50000,128],标准差=1.0/math.sqrt(embedding_size)
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))    # 初始化为全0

    # 定义所示函数和优化器
    # 计算该批次的平均NCE损失。
    # 每次我们评估损失时，tf.nn.nce_loss都会自动绘制一个新的负标签样本
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_biases,
                                         labels=train_labels, inputs=embed,
                                         num_sampled=num_sample,
                                         num_classes=vocabulary_size))
    # 采用随机梯队下降法
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 加入正则化，归一化
    # norm代表每一个词对应的词向量的长度矩阵
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
    normalized_embeddings = embeddings/norm     # normalized_embeddings就是每个词向量都除以自己的长度，即除以自己的模
                                                # normalized_embeddings本质是单位向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

    # 计算相似度，similarity就是valid_dataset 中对应的单位向量valid_embeddings与整个词嵌入字典中单位向量的夹角余弦
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # 添加变量初始化器
    initialization = tf.global_variables_initializer()


"""5.开始训练"""
num_steps = 100001      # 设置迭代次数十万零一次
with tf.Session(graph=graph) as sess:
    initialization.run()    # 如果没有上面的那句初始化，那么这句等价于sess.run(tf.global_variables_initializer())
    print('初始化完毕')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)    # 参数分别为128,2,1
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 每训练2000步就打印一次平均损失
        if step % 2000 ==0:
            if step > 0:
                average_loss /= 2000
            print('当训练到', step, '步时，平均损失是average_loss=', average_loss)
            average_loss = 0
        if step % 10000 ==0:
            sim = similarity.eval()    # 这就相当于sess.run()，就是运行一下得到相似度

            # 计算验证集余弦相似度最高的词
            for i in range(valid_size):
                # 根据id拿对应的单词
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                # 从大到小排序，不算自己本身，取前top_k个值
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s, %s:'% (log_str, close_word)
                print('log_str是：', log_str)
    # 训练结束，得到词向量
    final_embeddings = normalized_embeddings.eval()
    print('训练结束，得到最后的词向量：', final_embeddings)

"""6.可视化
降维后画出词向量"""
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.savefig(filename)   # 图片保存在该.py文件所在位置

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
   print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")