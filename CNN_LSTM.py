# encoding=utf-8
'''
CNN+LSTMㄎ語意分析模型
'''
from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb
import data_processing
import test_data_processing

# Embedding  词嵌入
max_features = 20000  # 词汇表大小
maxlen = 10  # 序列最大长度
embedding_size = 128  # 词向量维度

# Convolution  卷积
filter_length = 5  # 滤波器长度
nb_filter = 64  # 滤波器个数
pool_length = 4  # 池化长度

# LSTM
lstm_output_size = 70  # LSTM 层输出尺寸

# Training   训练参数
batch_size = 2  # 批数据量大小
nb_epoch = 5  # 迭代次数


# 载入模型
print('Loading data...')
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
(X_train, y_train)  =  data_processing.load_data()
(X_test,y_test)=test_data_processing.load_data()
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# 填充到固定长度 maxlen
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
# 构建模型
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))  # 词嵌入层
model.add(Dropout(0.25))  # Dropout层

# 1D 卷积层，对词嵌入层输出做卷积操作
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# 池化层
model.add(MaxPooling1D(pool_length=pool_length))
# LSTM 循环层
model.add(LSTM(lstm_output_size))
# 全连接层，只有一个神经元，输入是否为正面情感值
model.add(Dense(5))
model.add(Activation('softmax'))  # sigmoid判断情感

model.summary()  # 模型概述

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 训练
print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

# 测试
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

loss, accuracy = model.evaluate(X_test, y_test)
predict=np.round(model.predict(X_test))
print (model.predict(X_test))
print ("=====================")
print (model.predict_classes(X_test))
print ("=======================")
print (loss,accuracy)
