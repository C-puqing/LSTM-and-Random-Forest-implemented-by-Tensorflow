import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Dropout, Activation

feanum=1#一共有多少特征
window=5#时间窗设置
df1=pd.read_csv('/content/drive/MyDrive/data.csv') #读取数据
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
print(df.describe())

# 取16-19年训练，检测20年
dataset = []    
for i in range(5):
    dataset = dataset + list(df.iloc[:,i])

#这一部分在处理数据 将原始数据改造为LSTM网络的输入
stock=pd.DataFrame(dataset)
seq_len=window
amount_of_features = len(stock.columns)#有几列  1
data = pd.DataFrame(stock)
sequence_length = seq_len + 1#序列长度+1
result = []
for index in range(len(data) - sequence_length):#循环 数据长度-时间窗长度 次
    result.append(data[index: index + sequence_length])#第i行到i+5
result = np.array(result)#得到样本，样本形式为 window*feanum
cut=376#分训练集验证集，20年的做验证使用
train = result[:-cut, :]
x_train = train[:, :-1]
y_train = train[:, -1][:,-1]
x_valid = result[-cut:, :-1]
y_valid = result[-cut:, -1][:,-1]
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
X_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], amount_of_features))

#建立、训练模型过程
d = 0.10
model = Sequential()#建立层次模型
model.add(LSTM(16, input_shape=(window, feanum), return_sequences=True))#建立LSTM层
model.add(Dropout(d))#建立的遗忘层
model.add(LSTM(16, input_shape=(window, feanum), return_sequences=False))#建立LSTM层
model.add(Dropout(d))#建立的遗忘层
model.add(Dense(2,kernel_initializer='RandomUniform',activation='linear'))   #建立全连接层     
model.add(Dense(1,kernel_initializer='RandomUniform',activation='linear'))
model.compile(loss='mae',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size = 128) #训练模型nb_epoch次

#模型参数
# print(model.summary())

#在训练集上的拟合结果
y_train_predict=model.predict(X_train)[:,0]
y_train=y_train
draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[100:400,0].plot(figsize=(12,6))
draw.iloc[100:400,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题

#在验证集上的拟合结果
y_valid_predict=model.predict(X_valid)
draw=pd.concat([pd.DataFrame(y_valid),pd.DataFrame(y_valid_predict)],axis=1)
draw.iloc[100:400,0].plot(figsize=(12,6))
draw.iloc[100:400,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Valid Data",fontsize='30') #添加标题

#输出结果
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
print('训练集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_train_predict, y_train))
print(mean_squared_error(y_train_predict, y_train) )
print(mape(y_train_predict, y_train) )
print('测试集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_valid_predict, y_valid))
print(mean_squared_error(y_valid_predict, y_valid) )
print(mape(y_valid_predict, y_valid) )

#计算验证集的预测结果与真实值的精度误差
error = 0
for i in range(376):
    error += abs(y_valid[i] - y_valid_predict[i])
error /= 376
x = 1.5 #可控
print("error = ",error)
print("accuracy = ",(1 - error)/x *100)