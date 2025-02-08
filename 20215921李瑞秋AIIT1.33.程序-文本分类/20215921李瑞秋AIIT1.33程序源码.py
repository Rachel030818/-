import numpy as np
import jieba
import pandas as pd
import time
a = time.time()
# 提取词向量 
WV = {}
mean_vec = 0
with open(r'C:\Users\Rachel\Desktop\20215921李瑞秋AIIT1.33.程序-文本分类\20215921李瑞秋AIIT1.33.程序词向量.txt',encoding='utf-8') as f:
    for lines in f.readlines():
        w,v = lines.split("\t")
        WV[w] = np.array(eval(v))
        mean_vec += np.array(eval(v))
    f.close()
def W2v(w,WV,mean_vec):
    if w in WV.keys():
        return np.array(WV['w'])
    else:
        return mean_vec/len(WV)

# 各类数据的处理 - 分词,词向量化,特征和标签的分离
def datadeal(path):
    data = pd.read_csv(r"C:\Users\Rachel\Desktop\20215921李瑞秋AIIT1.33.程序-文本分类\20215921李瑞秋AIIT1.33程序数据\train.csv")
    data = pd.read_csv(path)
    data = data.drop('id',axis=1).values
    # print(type(data))
    for m in range(len(data)):
        for n in range(2):
            data[m,n] = jieba.cut_for_search(data[m,n])
    data1 = pd.DataFrame(data)
    data2 = data1.values

    for i in range(len(data2)):
        for z in range(2):
            X = np.zeros(100)
            for x in data2[i,z]:
                X += W2v(x,WV,mean_vec)
            data2[i,z] = X
    X =[]
    y = []
    for q in range(len(data2)):
        X.append(np.append(data2[q,0],data2[q,1]))
        y.append(data2[q,2])

    return np.array(X),np.array(y)

# 得到各类数据集的特征值和标签
X_train, y_train = datadeal(r"C:\Users\Rachel\Desktop\20215921李瑞秋AIIT1.33.程序-文本分类\20215921李瑞秋AIIT1.33程序数据\train.csv")
X_test, y_test = datadeal(r"C:\Users\Rachel\Desktop\20215921李瑞秋AIIT1.33.程序-文本分类\20215921李瑞秋AIIT1.33程序数据\test.csv")
X_dev, y_dev = datadeal(r"C:\Users\Rachel\Desktop\20215921李瑞秋AIIT1.33.程序-文本分类\20215921李瑞秋AIIT1.33程序数据\test.csv")
# print(X_train[0])
b = time.time()
print(b-a)

# 数据集分析

def match(label): 
    matching = 0
    for l in label:
        if l == 1:
            matching += 1
    return matching, matching/len(label)
matching_train ,train_matching = match(y_train) 
matching_test, test_matching = match(y_test)
matching_dev, dev_matching = match(y_dev)
print("训练集里匹配数量:", matching_train,"占比:",train_matching)
print("测试集里匹配数量:", matching_test,"占比:",test_matching)
print("验证集里匹配数量:", matching_dev,"占比:",dev_matching)


# 神经网络
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import tensorflow

classifier = Sequential(
    [Dense(activation = "relu", input_dim = 200,units = 100, kernel_initializer = "uniform"),
    Dense(activation = "relu", units = 50, kernel_initializer = "he_uniform"),
    Dense(activation = "sigmoid", units = 1,kernel_initializer = "he_uniform")
    ])
optimizer = Adam(learning_rate=0.1)
classifier.compile(optimizer = optimizer , loss = BinaryCrossentropy(), 
                   metrics = ['accuracy'] )
classifier.fit(X_train , y_train , batch_size = 600 ,epochs = 30, validation_data = (X_dev,y_dev), validation_freq= 2, class_weight={1:5,0:1.25} )
classifier.evaluate(X_dev)
print(X_test)
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.9)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
print("精度:",accuracy*100)

c = time.time()
print("神经网络模型时间:",c-b)
print("整个流程时间:",c-a)













