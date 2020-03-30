"""本代码是seq2seq的完整代码"""
from stanfordcorenlp import StanfordCoreNLP
"""第一步：对中英文数据进行切片"""
# 对数据切片
path = './data/'
en_path = path + 'train.raw.en'    # 这是未分词的文件地址
zh_path = path + 'train.raw.zh'    # 这是未分词的文件地址

seg_en_path = path + 'train.en.txt'    # 这是存储分好的词的文件地址
seg_zh_path = path + 'train.zh.txt'

train_en = open(seg_en_path, 'w', encoding='utf-8')
train_zh = open(seg_zh_path, 'w', encoding='utf-8')

nlp_zh = StanfordCoreNLP(r'E:\guozhentao-standfordcorenlp\stanford-corenlp', lang='zh')

with open(zh_path, mode='r', encoding='utf-8') as file:
    for word in file.readlines():
        if word != '\n':
            seg_word = nlp_zh.word_tokenize(word)      # seg_word是已经切好的词
            fenci_connect = ' '.join(seg_word)
            train_zh.write(fenci_connect+'\n')
        else:
            train_zh.write(word)
print('train_zh写入完毕！')
file.close()

nlp_en = StanfordCoreNLP(r'E:\guozhentao-standfordcorenlp\stanford-corenlp')    # 默认按照英文进行分词
with open(en_path, mode='r', encoding='utf-8') as f:
    for word in f.readlines():
        if word != '\n':
            seg_word_en = nlp_en.word_tokenize(word)      # seg_word是已经切好的词
            fenci_con = ' '.join(seg_word_en)
            train_en.write(fenci_con+'\n')
        else:
            train_en.write(word)
print('train_en写入完毕！')
f.close()