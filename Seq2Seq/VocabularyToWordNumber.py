""""""
"""将中英训练文件根据词汇文件转化为单词编号，
每个单词的编号就是这个单词在词汇表文件中的行号"""
import codecs
import sys

# 这两个是原始文件的地址，要把这两个原始已切好词的文件转化成单词编号文件
train_en_file_path = './data/train.en.txt'      # 训练数据英文文件
train_zh_file_path = './data/train.zh.txt'      # 训练数据中文文件

# 这两个是词汇表文件的地址
vocab_en_path = './data/train_vocab_en.txt'     # 这是BuildVocabularyFile.py生成的词汇表文件
vocab_zh_path = './data/train_vocab_zh.txt'     # 这是BuildVocabularyFile.py生成的词汇表文件

# 这两个用来存储编号文件
word_number_en_path = './data/train_word_number_en.txt'
word_number_zh_path = './data/train_word_number_zh.txt'


def word_to_number(path1, path2, path3):      # 参数path1是词汇表地址，path2是原始已切好词的文件地址，path3是存储单词编号的地址
    # 读取词汇文件，并建立词汇到单词编号的映射
    with codecs.open(path1, 'r', 'utf-8') as f_vocab:
        vocab = [line.strip() for line in f_vocab.readlines()]

        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        word_to_id = {k : v for (k, v) in zip(vocab, range(len(vocab)))}

    # 如果出现了被删除的低频次，则替换为'<unk>'的编号
    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

    fin = codecs.open(path2, 'r', 'utf-8')
    fout = codecs.open(path3, 'w', 'utf-8')
    for line in fin:
        words_list = line.strip().split() + ['<eos>']   # 读取单词，并添加'<eos>'结束符

        # 将每个单词替换成词汇表中的编号
        out_line = ' '.join([str(get_id(word)) for word in words_list]) + '\n'    # 每个单词取出对应的id后用空格连接
        fout.write(out_line)
    fin.close()
    fout.close()


if __name__ =='__main__':
    li = ['en', 'zh']
    for _ in li:
        if _ == 'en':
            word_to_number(path1=vocab_en_path, path2=train_en_file_path, path3=word_number_en_path)
        else:
            word_to_number(path1=vocab_zh_path, path2=train_zh_file_path, path3=word_number_zh_path)