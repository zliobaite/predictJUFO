#2013 11 15 I.Zliobaite
#analysis of JUFO rankings of jounrals based on title

import prepareDataJUFO
#from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from scipy.sparse import hstack

#parameter setting
file = 'lehdetjasarjat_13112013.csv'
#file = 'leh_small_example.csv'


#operation
titles,labels = prepareDataJUFO.extractNamesJUFO(file)
#print(titles)
print('number of journals',len(titles))

Xtrain,Xtest,ytrain,ytest = train_test_split(titles,labels,test_size=0.5,random_state=42)
#print(Xtrain)
Wtrain = [ len(x.split()) for x in Xtrain]
Wtest = [ len(x.split()) for x in Xtest]
#Ltrain = [ len(x) for x in Xtrain]
#Ltest = [ len(x) for x in Xtest]

#print(Wtest)

Xtrain, transformation = prepareDataJUFO.makeVectorsTrain(Xtrain)
#print(Xtrain)
print('words ',len(transformation.get_feature_names()))
Xtest = prepareDataJUFO.makeVectorsTest(Xtest,transformation)

Xtrain = hstack((Xtrain, np.matrix(Wtrain).T))
Xtest = hstack((Xtest, np.matrix(Wtest).T))

#Xtrain = hstack((Xtrain, np.matrix(Ltrain).T)) - makes things worse
#Xtest = hstack((Xtest, np.matrix(Ltest).T))

#Xtrain = np.matrix(Wtrain).T
#Xtest = np.matrix(Wtest).T

results = []
classifiers = [KNeighborsClassifier(3),
               KNeighborsClassifier(5),
               KNeighborsClassifier(7),
               #SVC(kernel="linear", C=0.025),
               #SVC(gamma=2, C=1),
               #DecisionTreeClassifier(max_depth=5),
               #RandomForestClassifier(max_depth=20, n_estimators=10),
               #AdaBoostClassifier(),
               #GaussianNB(),
               #LDA(),
               #QDA(),
               MultinomialNB()]
names = ["KNN(3)","KNN(5)","KNN(7)","MNB"]
i=0
for clf in classifiers:
    #clf = MultinomialNB()
    #MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    #print(ypred)
    acc = accuracy_score(ytest, ypred)
    #print('Accuracy ',acc)
    kp = prepareDataJUFO.kappa(ytest, ypred)
    #print('Kappa ',kp)
    print(names[i],'Acc ',acc,' kappa ',kp)
    results.append([acc,kp])
    i=i+1

print('<><><><>')
cm = confusion_matrix(ytest, ypred,labels=['-','1','2','3'])
print(cm)
#print(results)
