'''
目标：
练习降维之PCA
将iris datasets 四维数据做降维，实现二维平面的可视化
'''
#1. 导入packages
import matplotlib.pyplot as plt #加载matplotlib用于数据的可视化

from sklearn.decomposition import PCA #加载PCA算法包

from sklearn.datasets import load_iris #加载iris数据集导入函数


#2.加载数据并降维
data = load_iris() #以字典形式加载iris数据集
Y = data.target #Y表示数据集中的标签
X = data.data #X表示数据集中的数据属性

pca = PCA(n_components=2) #加载PCA算法，设置降维后主成分数目为2
reduced_X = pca.fit_transform(X)  #对原始数据进行降维，赋值保存在reduced_X中

#3.按类别对降维后的数据进行保存
red_x, red_y = [], [] #第一类数据点
blue_x, blue_y = [], [] # 第二类数据点
green_x, green_y = [], [] #第三类数据点


#4. 将降维后的数据点保存在不同的列表中
for i in range(len(reduced_X)):
    if Y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif Y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])


#5. 调用scatter()降维后数据点的可视化 
plt.scatter(red_x, red_y, c='r', marker='x')      #第一类数据点
plt.scatter(green_x, green_y, c='b', marker='D')  # 第二类数据点
plt.scatter(blue_x, blue_y, c='g', marker='.')    # 第三类数据点

plt.show() #可视化