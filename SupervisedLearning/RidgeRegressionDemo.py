'''
目的：练习岭回归及其应用实例
没有找到原始data
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


data = np.genfromtxt('data.txt')        #使用numpy的方法从txt文件中加载数据
plt.plot(data[:, 4])                    #使用plt展示车流量信息
X = data[:, :4]     #X用于保存0-3维数据，即属性
y = data[:, 4]      #y用于保存第4维数据，即车流量

poly = PolynomialFeatures(6)   #用于创建最高次数6次方的的多项式特征，多次试验后决定采用6次 
X = poly.fit_transform(X)      #X为创建的多项式特征
 
 
# 划分训练集和测试集:

# 将所有数据划分为训练集和测试集:train_set_, test_set_
# test_size表示测试集的比例 
# random_state是随机数种子 
train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, 
                                                                    test_size=0.3,
                                                                    random_state=0)

# 创建回归器，并进行训练:
clf = Ridge(alpha=1.0, fit_intercept=True)
clf.fit(train_set_X, test_set_y)        #调用fit函数使用训练集训练回归器
clf.score(test_set_X, test_set_y)

#画出一段200到300范围内的拟合曲线:
start = 200
end = 300
y_pre = clf.predict(X)          #调用predict函数的拟合值
time = np.arange(start, end)
plt.plot(time, y[start:end], 'b', label='real')
plt.plot(time, y_pre[start:end], 'r', label='predict')
plt.legend(loc='upper left')    #设置图例的位置
plt.show()

