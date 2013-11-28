import csv as csv
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import numpy as np
from stemming.porter2 import stem
import re
from random import shuffle

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
    return titles,rankings

def writeDataset(titles,labels):
    cwriter = csv.writer(open('JUFOdataset.csv', 'w'),delimiter=';')
    cwriter.writerow(["TITLE","RATING"])
    for sk in range(len(labels)):
        newLine = [titles[sk],labels[sk]]
        cwriter.writerow(newLine)

def extractNAmesPredatory(fileName):
    f = open(fileName)
    titles = f.readlines()
    f.close()
    rankings = ['-1'] * len(titles)
    return titles, rankings

def makeVectorsTrain(corpus):
    #corpus = cleanCorpus(corpus)
    #corpus = stemCorpus(corpus)
    vectorizer = CountVectorizer(max_features = 1000,strip_accents='unicode')
    #vectorizer = CountVectorizer(strip_accents='unicode')
    corpus = [cp.decode('latin-1').encode('utf-8') for cp in corpus]
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def numerLabels(y):
    sk = 0
    for tt in y:
        if tt =='-':
            y[sk] = '0'
        sk = sk + 1
    y = [int(tt) for tt in y]
    return y

def roundPredictions(y):
    y = [int(round(tt,0)) for tt in y]
    return y


def mae(ytest, ypred):
    ypred = numerLabels(ypred)
    ytest = numerLabels(ytest)
    #print(ypred)
    mae = mean_absolute_error(ytest,ypred)
    #randmae = mean_absolute_error(ytest,np.random.permutation(ytest))
    randnorm = mae/0.58
    return mae, randnorm

def makeVectorsTest(corpus,vectorizer):
    corpus = [cp.decode('latin-1').encode('utf-8') for cp in corpus]
    X = vectorizer.transform(corpus)
    return X

def kappa(ytrue,ypred):
    acc = accuracy_score(ytrue, ypred)
    cm = confusion_matrix(ytrue, ypred)
    priorPredictions = sum(cm)
    priorTrue = sum(cm.T)
    N = sum(priorTrue)
    priorTrue = 1.0*priorTrue/N
    print('priors',priorTrue)
    priorPredictions = 1.0*priorPredictions/N
    acc0 = sum(priorPredictions * priorTrue)
    kappa = (acc - acc0)/(1 - acc0)
    return kappa

def stemCorpus(corpus):
    #print(corpus[1])
    sk = 0
    #print(corpus)
    for row in corpus:
        #print(row)
        row = [stem(word) for word in row.split()]
        #print(row)
        corpus[sk] = ' '.join(row)
        sk = sk + 1
    #print(corpus[1])
    print('done stemming')
    return corpus

def cleanCorpus(corpus):
    sk = 0
    #print(corpus)
    for row in corpus:
        #print(row)
        row = [re.sub('^[^a-zA-z]*|[^a-zA-Z]*$','',word) for word in row.lower().replace('-',' ').split()]
        #print(row)
        corpus[sk] = ' '.join(row)
        sk = sk + 1
    #print(corpus[1])
    print('done cleaning')
    return corpus