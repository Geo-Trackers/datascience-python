from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier #Stochastic Gradient Descent SGD
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import numpy as np

# Input data
# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43],[160, 60, 38],[154, 54, 37],[166, 65, 40],[190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37],
[171, 75, 42], [181, 85, 43]]
# male or famle
Y = ['male','female','female','female','male','male','male','female','male','female','male']


# Initializing objects for each of the supervised learning models

clf = tree.DecisionTreeClassifier()
clf1 = SGDClassifier(loss = "log", penalty = "l2")
clf2 = GaussianNB()
clf3 = svm.SVC()
clf4 = neighbors.KNeighborsClassifier()


# Fitting the models

clf = clf.fit(X,Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
clf4 = clf4.fit(X, Y)


# Checking the accuracy of decision tree
prediction = clf.predict(X)
acc_decision_tree = accuracy_score(prediction,Y)*100


#Checking the accuracy of Stochastic Gradient Descent

prediction1 = clf1.predict(X)
acc_SGD = accuracy_score(prediction1,Y)* 100

#Checking the accuracy of Gaussian NB
prediction2 = clf2.predict(X)
acc_GNB = accuracy_score(prediction2, Y)*100


#Checking the accuracy of Support Vector Machines
prediction3 = clf3.predict(X)
acc_SVM = accuracy_score(prediction3, Y)*100


#Checking the accuracy of KNeighbors
prediction4 = clf4.predict(X)
acc_KN = accuracy_score(prediction4, Y)*100


#Determining the best classifier among all

accurate = np.argmax([acc_decision_tree,acc_SGD, acc_GNB, acc_SVM, acc_KN])
classification = {0:'Decision_Tree', 1:'Stochastic_Gradient_Descent', 2: 'GaussianNB', 3:'SVM', 4:'KNeighbors'}
print ('Accuracy for the decision tree:', acc_decision_tree)
print ('Accuracy of Stochastic-Gradient-Descent:', acc_SGD)
print ("Accuracy of Gaussian NB : ", acc_GNB)
print ("Accuracy of SVM :", acc_SVM)
print ('Accuracy of KNeighbors: ', acc_KN)


print ('The most accurate gender classifier is:', classification[accurate])
print ('Thank you ')
