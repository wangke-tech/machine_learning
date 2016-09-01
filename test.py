#!/usr/bin/env python
# encoding: utf-8

import os
import sys
HOME_PATH = os.path.dirname(os.path.abspath(__file__))
BIN_PATH = HOME_PATH + '/bin'
TXT_PATH = HOME_PATH + '/data/points.txt'
sys.path.append(BIN_PATH)

from numpy import *
import time
from kmeans import *

## step1: load data
dataSet = []
with open(TXT_PATH,'r') as r:
    for line in r.readlines():
        lineArr = line.strip().split('   ')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

# step2: clustering
dataSet = mat(dataSet)
k =4
centroids, clusterAssment = kmeans(dataSet, k)

# step3: show the result
print 'show the result...'
showCluster(dataSet, k, centroids, clusterAssment)



