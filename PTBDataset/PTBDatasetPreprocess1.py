"""此代码"""
"""用于把PTB数据集预处理成一个词汇表文件
文件的TXT文本，通过编辑器查看，里面一个单词一行"""

import collections
import codecs   # codecs专门用作编码转换
from operator import itemgetter
train_data = 'E:\\guozhentao-python\\PythonNLP\\data\\ptb.train.txt'    # 训练集数据文件的地址
vocabulary_output = './data/ptb.vocabulary.txt'     # 输出的词汇表文件

counter = collections.Counter()     # 统计单词出现的频率
with codecs.open(train_data, 'r', 'utf-8') as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

# 按照词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
print('sorted_word_to_cnt是', sorted_word_to_cnt)
sorted_words = [x[0] for x in sorted_word_to_cnt]
print('sorted_words是', sorted_words)

with codecs.open(vocabulary_output, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')       # 至此，通过以上程序可以创建一个词汇表文件ptb.vocabulary.txt
    print('创建词汇文件成功！')