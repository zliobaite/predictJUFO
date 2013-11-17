import csv as csv
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from sklearn.metrics import accuracy_score
import numpy as np

def extractNamesJUFO(fileName):
    csvFileObject = csv.reader(open(fileName, 'rU'),delimiter=";")
    header = csvFileObject.next()
    hi = header.index(' NIMEKE/TITLE')
    hi2 = header.index(' TASO/LEVEL')
    rankings = []
    titles = []
    for row in csvFileObject:
        rankings.append(row[hi2])
        titles.append(row[hi])
    #print(rankings[5])
    #print(titles[5])
    return titles,rankings

def makeVectorsTrain(corpus):
    vectorizer = CountVectorizer(min_df=0.0006)
    corpus = [cp.decode('latin-1').encode('utf-8') for cp in corpus]
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def makeVectorsTest(corpus,vectorizer):
    corpus = [cp.decode('latin-1').encode('utf-8') for cp in corpus]
    X = vectorizer.transform(corpus)
    return X

def kappa(ytrue,ypred):
    acc = accuracy_score(ytrue, ypred)
    ypred = np.random.permutation(ypred)
    acc0 = accuracy_score(ytrue, ypred)
    kappa = (acc - acc0)/(1 - acc0)
    return kappa