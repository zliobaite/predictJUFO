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
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report, mean_absolute_error
from scipy.sparse import hstack
from sklearn.feature_selection import SelectKBest,chi2
from sklearn import linear_model, svm

#parameter setting
file = 'lehdetjasarjat_13112013.csv'
#file = 'leh_small_example.csv'
file2 = 'predatory_journals.txt'

#operation
titles,labels = prepareDataJUFO.extractNamesJUFO(file)

#predatory journals
titles2,labels2 = prepareDataJUFO.extractNAmesPredatory(file2)
titles = titles + titles2
labels = labels + labels2
labels = prepareDataJUFO.numerLabels(labels)

#prepareDataJUFO.writeDataset(titles,labels)

print('number of journals',len(titles))

#extra features
Xtrain,Xtest,ytrain,ytest = train_test_split(titles,labels,test_size=0.5,random_state=42)
Wtrain = [ len(x.split()) for x in Xtrain]
Wtest = [ len(x.split()) for x in Xtest]
Ltrain = [ len(x) for x in Xtrain]
Ltest = [ len(x) for x in Xtest]
ALtrain = [ np.mean( len(x.split()) ) for x in Xtrain]
ALtest = [ np.mean( len(x.split()) ) for x in Xtest]

Xtrain, transformation = prepareDataJUFO.makeVectorsTrain(Xtrain)
print('words ',len(transformation.get_feature_names()))
Xtest = prepareDataJUFO.makeVectorsTest(Xtest,transformation)

#add extra features
Xtrain = hstack((Xtrain, np.matrix(Wtrain).T))
Xtest = hstack((Xtest, np.matrix(Wtest).T))
Xtrain = hstack((Xtrain, np.matrix(Ltrain).T))
Xtest = hstack((Xtest, np.matrix(Ltest).T))
Xtrain = hstack((Xtrain, np.matrix(ALtrain).T))
Xtest = hstack((Xtest, np.matrix(ALtest).T))

featureSelector = SelectKBest(chi2, 300)
Xtrain = featureSelector.fit_transform(Xtrain, ytrain)
Xtest = featureSelector.transform(Xtest)
#print(featureSelector.get_support())


results = []
classifiers = [KNeighborsClassifier(3),
               MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
               BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)]
names = ["KNN(3)","MNB","BernNB"]
i=0
for clf in classifiers:
    clf.fit(Xtrain, ytrain)
    ypred = clf.predict(Xtest)
    acc = accuracy_score(ytest, ypred)
    mae, rmae = prepareDataJUFO.mae(ytest, ypred)
    kp = prepareDataJUFO.kappa(ytest, ypred)
    print(names[i],'Acc ',acc,' kappa ',kp, ' mae ',mae, ' rand normalized ',rmae)
    results.append([acc,kp])
    i=i+1
    #cm = confusion_matrix(ytest, ypred,labels=['-1','-','1','2','3'])
    #print(cm)
    print classification_report(ytest, ypred)

probabilities = clf.feature_log_prob_
probabilities = np.exp(probabilities)
colSums = probabilities.sum(axis=0)
probabilities = probabilities / colSums[np.newaxis,:]
print(probabilities)

#analysis of selected features
indSorted   = np.argsort(featureSelector.scores_)[::-1]
print(indSorted[0:30])
labelsList = list(transformation.vocabulary_) + list(["#words"]) + list(["#symbols"]) + list(["av.symb.per.word"])
for sk in range(20):
    indNow = indSorted[sk]
    out = "\t"+labelsList[indNow]+"\t\t"+str(np.round(100*probabilities[0,sk])).strip('.0')+"%\t\t"+str(np.round(100*probabilities[1,sk])).strip('.0')+"%\t\t"+str(np.round(100*probabilities[2,sk])).strip('.0')+"%\t\t"+str(np.round(100*probabilities[3,sk])).strip('.0')+"%\t\t"+str(np.round(100*probabilities[4,sk])).strip('.0')+"%"
    print(out)

clf.fit(Xtrain, prepareDataJUFO.numerLabels(ytrain))
ypred = prepareDataJUFO.roundPredictions(clf.predict(Xtest))
mae = mean_absolute_error(prepareDataJUFO.numerLabels(ytest),prepareDataJUFO.numerLabels(ypred))
print('regression mae',mae)