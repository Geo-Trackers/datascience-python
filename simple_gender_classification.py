from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier #Stochastic Gradient Descent SGD
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors

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

#Predicting

prediction = clf.predict([[190, 70, 43]])
prediction1 = clf1.predict([[190,70,43]])
prediction2 = clf2.predict([[190,70,43]])
prediction3 = clf3.predict([[190,70,43]])
prediction4 = clf4.predict([[190,70,43]])

print ("Decision_Tree:",prediction)
print ("Stochastic Gradient Descent:", prediction1)
print ("Gaussian NB:", prediction2)
print ("Support Vector Machines:", prediction3)
print ("KNeighborsClassifier:", prediction4)

accuracy = max(prediction, prediction1, prediction2, prediction3, prediction4)

if accuracy == prediction :
    print("The most accurate model is: Decision tree")
elif accuracy == prediction1 :
    print("The most accurate model is: Stochastic Gradient Descents")
elif accuracy == prediction2 :
    print("The most accurate model is: Gaussian NB")
elif accuracy == prediction3 :
    print("The most accurate model is: Support Vector Machines.SVC")
elif accuracy == prediction4 :
    print ("The most accurate model is: KNeighbors Classifiers")
