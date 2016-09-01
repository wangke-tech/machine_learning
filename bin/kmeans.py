#!/usr/bin/env python
# encoding: utf-8

from numpy import *
import matplotlib.pyplot as plt

# get euclide distance
def euclDistance(vector1, vector2):
    return sqrt(sum(square(vector1 - vector2)))

def initCentroids(dataSet, k):
    sampleNum, dim = dataSet.shape
    centroids = mat(zeros((k, dim)))
    for i in range(k):
        index = random.uniform(0, sampleNum)
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    samplesNum = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between the sample and tis cenroid
    clusterAssment = mat(zeros((samplesNum, 2)))
    clusterChanged = True

    ## step1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in xrange(samplesNum):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance <minDist:
                    minDist = distance
                    minIndex = j

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist **2
    ## step 4: update centroids
    for j in range(k):
        pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
        centroids[j, :] = mean(pointsInCluster, axis=0)

    print 'Congratulations, cluster completed!'
    return centroids, clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print 'Sorry! I can not draw because the dimension of your data is not 2'
        return

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'dr', '<r', 'pr']
    if k > len(mark):
        print 'Sorry! Your k is too large!'
        return

    # draw all samples
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)


    plt.show()
