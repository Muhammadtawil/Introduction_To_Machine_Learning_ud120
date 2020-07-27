#!/usr/bin/python

import sklearn
import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
print(len(features_train))

#output: 150

from sklearn import tree 
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred,labels_test)
print(acc)
print('Accuracy on test set = {0}'.format(clf.score(features_test, labels_test)))   
#0.947667804323


# 识别最强大特征
# 选择（过拟合）决策树并使用 feature_importances_ 属性来获得一个列表， 
# 其中列出了所有用到的特征的相对重要性（由于是文本数据，因此列表会很长）。 
# 我们建议迭代此列表并且仅在超过阈值（比如 0.2——记住，所有单词都同等重要，每个单词的重要性都低于 0.01）
# 的情况下将特征重要性打印出来。

imp = clf.feature_importances_
print(max(imp))   
# 0.764705882353

print(imp.argmax()) 
#33614


#使用 TfIdf 获得最重要的单词
# 为了确定是什么单词导致了问题的发生，你需要返回至 TfIdf，使用你从迷你项目的上一部分中获得的特征数量来获取关联词。 
# 你可以在 TfIdf 中调用 get_feature_names() 来返回包含所有单词的列表； 抽出造成大多数决策树歧视的单词。

# 这个单词是什么？类似于签名这种与 Chris Germany 或 Sara Shackleton 唯一关联的单词是否讲得通？

# what's the most powerful word when your decision tree is making its classification decisions?
# 当你的决策树进行分类决策时，最强大的词是什么？

# 请确保你修改 find_signature.py 以获得最有影响力的单词。

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

words_bag = vectorizer.get_feature_names()
print(words_bag[33614])  
#sshacklensf

# top_features = [(number, feature, vectorizer.get_feature_names()[number]) for number, feature in 
#                 zip(range(len(clf.feature_importances_)), clf.feature_importances_) if feature > 0.2]
# print(top_features)
# output: [(33614, 0.76470588235294124, u'sshacklensf')]



# 删除、重复
# 从某种意义上说，这一单词sshacklensf看起来像是一个异常值，所以让我们在删除它之后重新拟合。 
# 返回至 text_learning/vectorize_text.py，使用我们删除“sara”、“chris”等的方法，从邮件中删除此单词。 
# 重新运行 vectorize_text.py，完成以后立即重新运行 find_signature.py

words = words.replace("sara",'').replace("shackleton",'').replace("chris",'')\
             .replace("germani",'').replace("sshacklensf",'')

from sklearn import tree 
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred,labels_test)

imp = clf.feature_importances_
for index,feature in enumerate(imp):
    if feature >0.2:
        print(index,feature)  
#14343 0.666666666667


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

words_bag = vectorizer.get_feature_names()
print(words_bag[14343])   
#cgermannsf


# 再次检查重要特征
# 再次更新 vectorize_test.py 后重新运行。然后，再次运行 find_signature.py

words = words.replace("sara",'').replace("shackleton",'').replace("chris",'')\
                .replace("germani",'').replace("sshacklensf",'').replace("cgermannsf",'')
                
from sklearn import tree 
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred,labels_test)  #0.816837315131

imp_list=[]
imp = clf.feature_importances_
for index,feature in enumerate(imp):
    if feature >0.2:
        imp_list.append(feature)
        print(index,feature)  #21323 0.363636363636
        print(len(imp_list))  #1


# remove the 2 outlier words
sw = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]
sara_and_chris(sw)

# re-fit the tree
words_file = 'your_word_data.pkl'
authors_file = 'your_email_authors.pkl'
(clf, vectorizer, features_train, features_test, labels_train, labels_test) = my_func(words_file, authors_file)

print('Accuracy on test set = {0}'.format(clf.score(features_test, labels_test)))
