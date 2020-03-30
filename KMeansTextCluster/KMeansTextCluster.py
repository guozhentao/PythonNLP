"""使用Kmeans进行文本聚类"""
"""大体分以下几步：
   文本处理：分词、去停用词
   构建词向量
   kmeans聚类"""

import zipfile
import jieba
import re
import collections
from operator import itemgetter

# 1.读取需要聚类的数据

def read_zip_data(path):    # 用来读取压缩数据TextClusterData.zip
    with zipfile.ZipFile(path) as f:    # zipfile.ZipFile()就是解压缩文件
        data_list = []
        for i in range(93):    # 这个压缩文件一共有93个文件
            data = f.read(f.namelist()[i])
            seg_data_list = (' '.join(jieba.cut(data))).split()
            data_list.extend(seg_data_list)

        data_set = set(data_list)   # 这是对data_list去重,去重后转成了set对象
        data_set_to_list = list(data_set)
    return data_set_to_list     # 这是去完重的93篇文档的总的分词结果，但未去停用词


zip_file_path = 'E:\\guozhentao-python\\PythonNLP\\data\\KmeansTextClusterData.zip'
file_data = read_zip_data(zip_file_path)    # 这个file_data是列表的形式，到这一步，已去重，但尚未去停用词
print('file_data的内容是：', file_data)
print('file_data列表的长度是：', len(file_data))


def read_file(file_path):    # 用来读取停用词文件,参数是停用词文件路径
    with open(file_path, 'r') as f:
        words = f.read()
        seg_list = ' '.join(jieba.cut(words))
        stop_words_list = seg_list.split()
    return stop_words_list   # 返回的是停用词list


def del_stop_words(words, stop_words_file):     # 参数words代表已经分好词的待去除停用词的文本数据
    stop_words_list = read_file(stop_words_file)
    new_words = []
    for word in words:
        if word not in stop_words_list:
            value = re.compile(r'^[\u4e00-\u9fa5]{2,}$')    # 只匹配中文2字词及以上
            if value.match(word):
                new_words.append(word)
    return new_words


stop_words_file_path = 'E:\\guozhentao-python\\PythonNLP\\data\\new_stop_words.txt'
vocabulary = del_stop_words(file_data, stop_words_file_path)    # 到这获得的vocabulary是一个去完重的去完停用词的总的词结果
print('vocabulary为', vocabulary)
print('vocabulary的长度是', len(vocabulary))
# 以上获得了总数据，即全部文档的内容分好词后放在一个list后去重去停用词后的总结果


# 下面解析出每一篇文档的内容，组成一个list，然后放到一个大list中，成为形如[[], [], []]
def get_doc_content_list(path):
    with zipfile.ZipFile(path) as f:
        doc_content_list = []
        for i in range(93):
            data = f.read(f.namelist()[i])
            seg_data_list = (' '.join(jieba.cut(data))).split()
            data_list = del_stop_words(seg_data_list, stop_words_file_path)
            doc_content_list.append(data_list)
    return doc_content_list


doc_content_list = get_doc_content_list(zip_file_path)
print('doc_content_list', doc_content_list)
print('doc_content_list的长度是', len(doc_content_list))


# 2.构建向量空间模型（VSM，vector space model)，本质是词袋模型
counter = collections.Counter()
