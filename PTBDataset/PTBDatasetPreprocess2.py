""""""
"""该代码把训练文件、测试文件等根据词汇文件转化为单词编号，每个单词的编号
就是它在词汇文件中的行号"""

import codecs
import sys
train_data = 'E:\\guozhentao-python\\PythonNLP\\data\\ptb.train.txt'    # 这是原始训练集数据文件的地址
vocabulary_file = './data/ptb.vocabulary.txt'     # 这是已经存在的词汇表文件
data_output = './data/ptb_train_id.txt'     # 将单词替换为单词编号后的输出文件
with codecs.open(vocabulary_file, 'r', 'utf-8') as f_vocab:     # 读取词汇表，并建立词汇到单词编号的映射
    vocabulary = [w.strip() for w in f_vocab.readlines()]       # vocabulary是一个字符串list
    print('vocabulary为', vocabulary)

word_to_id = {k: v for (k, v) in zip(vocabulary, range(len(vocabulary)))}
print('word_to_id是', word_to_id)

# 如果出现了被删除的低频词，则替换为'<unk>'
# 给予一个单词，如果这个单词在字典word_to_id内，则返回单词对应的编号，如果不在字典内，则返回unk对应的编号
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id['<unk>']


fin = codecs.open(train_data, 'r', 'utf-8')
fout = codecs.open(data_output, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ['<eos>']    # 读取单词并添加<eos>结束符
    print('words是', words)
    # 将每个单词替换成词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    print('out_line是', out_line)
    fout.write(out_line)

fin.close()
fout.close()

