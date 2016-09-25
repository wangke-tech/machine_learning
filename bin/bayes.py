#!/usr/bin/env python
# encoding: utf-8


def trainNB0(trainMatrix, trainCategory):
    from numpy import *
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num, p1Num = zeros(numWords),zeros(numWords)
    p0Denom, p1Denom = 0.0, 0.0
    for i in range(numTrainDocs):
        if 1==trainCategory[i]:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect, p0Vect = p1Num /p1Denom, p0Num /p0Denom
    return p0Vect, p1Vect, pAbusive

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


def setOfWords2Vec(vecList, dataInput):
    ret = [0] * len(vecList)
    for word in dataInput:
        if word in vecList:
            ret[vecList.index(word)] = 1
        else:
            print "the word %s is not in my Vocabulary !" %(word,)
    return ret

if '__main__' == __name__:
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    print p0V,'\n', p1V, '\n', pAb
