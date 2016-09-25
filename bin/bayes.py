#!/usr/bin/env python
# encoding: utf-8
from numpy import *


# step1: load data
def loadDataSet():
    postingList =[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]   # 0=正常言论 1=侮辱性文字
    return postingList, classVec

def createVocabList(dataSet):
    retVec = set()
    for document in dataSet:
        retVec = retVec | set(document)
    return list(retVec)


# step2: convert words to vector
def setOfWords2Vec(vecList, dataInput):
    ret = [0] * len(vecList)
    for word in dataInput:
        if word in vecList:
            ret[vecList.index(word)] = 1
        else:
            print "the word %s is not in my Vocabulary !" %(word,)
    return ret


# step3: train
def trainNB0(trainMatrix, trainCategory):

    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(numTrainDocs)

    p0Num, p1Num = ones(numWords), ones(numWords)
    p0Denom, p1Denom = 2.0, 2.0

    for i in range(numTrainDocs):
        if 1==trainCategory[i]:
            p1Num += trainMatrix[i]   # 分子: 拼接向量
            p1Denom += sum(trainMatrix[i])   # 分母: 求和
        else:
            p0Num += trainMatrix[i]   # 分子: 拼接向量
            p0Denom += sum(trainMatrix[i])   # 分母: 求和

    p1Vect, p0Vect = log(p1Num /p1Denom), log(p0Num /p0Denom)

    return p0Vect, p1Vect, pAbusive


# step4: classify
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    return 1 if p1 > p0 else 0


# step5: test
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)

    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb)

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classifed as : ', classifyNB(thisDoc, p0V, p1V, pAb)


if '__main__' == __name__:
    testingNB()
