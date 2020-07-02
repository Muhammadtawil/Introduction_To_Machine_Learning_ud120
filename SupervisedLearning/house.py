'''
目的：线性回归+房价与房屋尺寸关系的线性拟合
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# 2.加载训练数据，建立回归方程

datasets_X = []     # 建立datasets_X,用来存储数据中的房屋尺寸
datasets_Y = []     # 建立datasets_Y,用来存储数据中的房屋成交价格

fr = open('price.txt', 'r')     # 打开数据集所在文件
lines = fr.readlines()          # 一次读取整个文件

for line in lines:                      # 逐行进行操作，循环遍历所有数据
    items = line.strip().split(',')     # 去除数据文件中的逗号  
    #items = [x for x in items if x.strip() != ''] 
    #print(items[0],items[1])  
    datasets_X.append(int(items[0]))    # 将读取的数据转换为int型，并分别写入datasets_X和datasets_Y
    datasets_Y.append(int(items[1]))

length = len(datasets_X)                                # 求得datasets_X的长度，即为数据的总数
datasets_X = np.array(datasets_X).reshape([length, 1])  # 将datasets_X转化为数组，
                                                        # 并变为二维，以符合线性回归拟合函数输入参数要求。

datasets_Y = np.array(datasets_Y)                       # 将datasets_Y转 化为数组



# 以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图。
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX, maxX).reshape([-1, 1])

# 调用线性回归模块，建立回归方程，拟合数据
linear = linear_model.LinearRegression()
linear.fit(datasets_X, datasets_Y)


print('Coefficients:', linear.coef_)        #查看回归方程系数
print('interept:', linear.intercept_)       #查看回归方程截距


#3. 可视化处理
plt.scatter(datasets_X, datasets_Y, color = 'red',label='origin data')      # scatter函数用于绘制数据点
plt.plot(X, linear.predict(X), color = 'blue',label='linear regression prediction')  # plot函数用来绘制直线
plt.legend()    #使label生效
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()