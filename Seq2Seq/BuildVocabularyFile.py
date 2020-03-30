"""说明："""
"""创建词汇表文件"""
"""Datasegmentation.py已经对原始文件进行了分词处理，返回的结果文件是
train.en.txt和train.zh.txt。
本代码就是对这两个文件的再次进行数据预处理，以便seq2seq模型使用"""

import codecs
import collections
from operator import itemgetter
train_en_file_path = './data/train.en.txt'      # 训练数据英文文件
train_zh_file_path = './data/train.zh.txt'      # 训练数据中文文件

vocab_output_en = './data/train_vocab_en.txt'
vocab_output_zh = './data/train_vocab_zh.txt'

def deal(lang):
    if lang == 'en':
        return deal_data(path1=train_en_file_path, path2=vocab_output_en)
    if lang == 'zh':
        return deal_data(path1=train_zh_file_path, path2=vocab_output_zh)


def deal_data(path1, path2):
    counter = collections.Counter()
    with codecs.open(path1, 'r', 'utf-8') as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1      # 统计词频

    # 按照词频顺序对单词进行降序排序
    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
    print('sorted_word_to_cnt是', sorted_word_to_cnt)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 把句子起始符'<sos>'，低频词符'<unk>'，句子结束符'<eos>'加入到词汇表
    sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words
    if len(sorted_words) > 10000:
        sorted_words = sorted_words[:10000]

    with codecs.open(path2, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word+'\n')
        print('成功创建词汇表文件！')


if __name__ == '__main__':
    en_zh_parameter = ['en', 'zh']
    for parameter in en_zh_parameter:
        deal(parameter)