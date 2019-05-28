#encoding=utf-8
import numpy as np
import re
import itertools
from collections import Counter
import jieba.analyse
import csv
import unicodecsv
import codecs
def load_data_and_labels():
    a=[]
    #x_text =list( open('example.csv', 'rb'))
    #載入data
    year_data = list(open('year_test_data', 'rb').readlines())
    who_data = list(open('who_test_data', 'rb').readlines())
    height_data = list(open('height_test_data', 'rb').readlines())
    class_4 = list(open('體重_test', 'rb').readlines())
    class_5 = list(open('住家_test', 'rb').readlines())
    #class_6 = list(open('住家環境_test', 'rb').readlines())
    #class_7 = list(open('職業_test', 'rb').readlines())
    x_train_text = year_data + who_data + height_data + class_4+class_5
    #x_train_text = year_data + who_data + height_data + class_4+class_5

    # 分類標籤＿訓練
    year_labels = [[1, 0, 0, 0,0] for _ in year_data]
    who_labels = [[0, 1, 0, 0,0] for _ in who_data]
    height_labels = [[0, 0, 1, 0,0] for _ in height_data]
    class_4_labels = [[0, 0, 0, 1,0] for _ in class_4]
    class_5_labels = [[0, 0, 0, 0, 1] for _ in class_5]
    #class_6_labels = [[0, 0, 0, 0, 0, 1, 0] for _ in class_6]
    #class_7_labels = [[0, 0, 0, 0, 0, 0, 1] for _ in class_6]
    y_train = np.concatenate([year_labels, who_labels, height_labels, class_4_labels,class_5_labels], 0)
    #y_train = np.concatenate([year_labels, who_labels, height_labels, class_4_labels, class_5_labels],0)

    # 定冠詞去除(你我他是的了）
    noise_list0 = [u'\u4f60', u'\u6211', u'\u4ed6', u'\u662f', u'\u4e86', '\n', u'?', u'\u4e86']
    #結巴斷詞
    for i in x_train_text:
        i = list(jieba.cut(i, cut_all=False))
        for j in i:
            if j in noise_list0:
                i.remove(j)
        a.append(i)

    # 寫入csv擋
    '''f = open('data.csv', 'w')
    f.write(codecs.BOM_UTF8)
    for i in x_text:
        i.decode('utf-8')
        f.write(i)
    f.close()'''
    return [a,y_train]

def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

#def build_vocab(sentences):
def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    sentences, labels = load_data_and_labels()
    #sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences)
    x, y = build_input_data(sentences, labels, vocabulary)
    return (x,y)








