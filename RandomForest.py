import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

feanum=1#一共有多少特征
window=5#时间窗设置

path = '/content/drive/MyDrive/data.csv' #文件路径
df1=pd.read_csv(path) #读取数据

min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)

# 取16-19年训练，检测20年
dataset = [] #数据集合
for i in range(5):
    dataset = dataset + list(df.iloc[:,i])

#处理数据
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
x_test = result[-cut:, :-1]
y_test = result[-cut:, -1][:,-1]
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

#更改数组shape
X_train=X_train.reshape(len(X_train),window)
y_train=y_train.reshape(len(X_train))
X_test=X_test.reshape(cut,window)
y_test=y_test.reshape(cut)

# Fit regression model
rf=RandomForestRegressor()
model = rf.fit(X_train, y_train)

#在训练集上的拟合结果
y_train_predict=model.predict(X_train)

#展示在训练集上的表现
draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[100:400,0].plot(figsize=(12,6))
draw.iloc[100:400,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题

#在测试集上的预测
y_test_predict=model.predict(X_test)

#展示在测试集上的表现
draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[:,0].plot(figsize=(12,6))
draw.iloc[:,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题

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
print(mean_absolute_error(y_test_predict, y_test))
print(mean_squared_error(y_test_predict, y_test) )
print(mape(y_test_predict, y_test) )

#计算验证集的预测结果与真实值的精度误差
error = 0
for i in range(376):
    error += abs(y_test[i] - y_test_predict[i])
error /= 376
x = 1.5 #可控
print("error = ",error)
print("accuracy = ",(1 - error)/x *100)