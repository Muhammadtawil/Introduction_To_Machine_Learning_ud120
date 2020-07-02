'''
目的：
练习 KMeans算法， 实现图片分割
'''


import numpy as np
import PIL.Image as image           #加载创建图片
from sklearn.cluster import KMeans  #加载KMeans算法


#加载训练数据集
def loadData(filepath):
    f = open(filepath, 'rb')        #以二进制方式打开文件
    data = []
    img = image.open(f)             #以列表形式返回图片像素值
    m,n = img.size                  # 获取图片的大小
    
    for i in range(m):              #将每个像素点RGB颜色处理到0-1的范围内，并存进data[]
        for j in range(n):
            x,y,z = img.getpixel((i, j))
            data.append([x/256.0, y/256.0, z/256.0])
    f.close()
    return np.mat(data), m, n        #以矩阵形式返回data，以及图片大小
imgData, row, col = loadData('bull.jpg') #加载数据


#加载Kmeans聚类方法
'''
不同的k值，聚类结果不同，需要自行摸索尝试调整k值
'''
km = KMeans(n_clusters = 3)         #制定3个聚类中心
#km = KMeans(n_clusters = 5)         #制定5个聚类中心


#对像素点进行聚类并输出

label = km.fit_predict(imgData)         #获取每个像素点所属的类别
label = label.reshape([row, col])       #reshape

pic_new = image.new("L", (row, col))    #创建一张新的灰度图保存聚类后的结构


#根据所属类别添加灰度值：
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256/(label[i][j] + 1)))

pic_new.save("KMeansResult.jpg", "JPEG")
#pic_new.save("KMeansResult2.jpg", "JPEG") ###km = KMeans(n_clusters = 5) 