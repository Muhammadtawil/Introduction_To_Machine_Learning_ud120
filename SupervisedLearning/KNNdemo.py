'''
目的：
练习KNN近邻分类器的使用方法
'''

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3) 
neigh.fit(X, y)

print(neigh.predict([[1.1]]))