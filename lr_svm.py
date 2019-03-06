#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import numpy as np


# In[2]:
trainingDataFilename = sys.argv[1]
testDataFilename = sys.argv[2]
modelIdx = int(sys.argv[3])

#trainingDataFilename = 'trainingSet.csv'
#testDataFilename = 'testSet.csv'
#modelIdx = 1


# In[3]:


def sigmoid(scores):
    predictions = np.zeros(len(scores))
    for i in range(len(predictions)):
        if scores[i] >= 0:
            predictions[i] +=  1.0 / (1.0 + np.exp(-scores[i]))
        else:
            predictions[i] += np.exp(scores[i]) / (1.0 + np.exp(scores[i]))
    return predictions

def lr(trainingSet, testSet):
    #print len(trainingSet.columns)
    regularization = 0.01
    step_size = 0.01
    
    max_iterations = 500
    tol = 1e-6
    
    count = 0
    
    train_labels = trainingSet['decision']    
    trainingSet = trainingSet.drop('decision', axis=1)
    
    #print train_labels, trainingSet
    w = np.zeros(len(trainingSet.columns) + 1)
    
    # Add intercept
    X = np.array(trainingSet)
    Y = np.array(train_labels)
    intercept = np.ones((X.shape[0], 1))
    #X = np.concatenate((X, intercept.T), axis=1)
    X = np.hstack((X, intercept))
    diff = 100.0
    
    while(count < max_iterations and diff > tol):
        count += 1
        norm_old = np.linalg.norm(w)
        
        scores = np.dot(X, w)
        predictions = sigmoid(scores)

        gradient = np.dot(X.T, (predictions - Y))

        for j in range(len(w)):
            gradient[j] += regularization * w[j]
            
        #gradient /= len(train_labels)
        w -= step_size * gradient
        norm_new = np.linalg.norm(w)
        
        diff = abs(norm_new - norm_old)
        #print w
        #print count, diff
    
    return w


# In[4]:


def svm(trainingSet, testSet):
    #print len(trainingSet.columns)
    regularization = 0.01
    step_size = 0.50
    
    max_iterations = 500
    tol = 1e-6
    #print len(trainingSet[trainingSet['decision'] == 1])
    count = 0
    train_labels = trainingSet['decision']    
    trainingSet = trainingSet.drop('decision', axis=1)

    w = np.zeros(len(trainingSet.columns) + 1)
    
    # Add intercept
    X = np.array(trainingSet)
    Y = np.array(train_labels)
    #print train_labels
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1.0
        else:
            Y[i] = 1.0
    #print Y.tolist()
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((intercept, X))
    diff = 100.0
    while(count < max_iterations and diff > tol):
        count += 1
        norm_old = np.linalg.norm(w)
        
        predictions = np.dot(X, w)
        #print X.shape, w.shape,predictions.shape
#         p = []
#         for i in range(0, 5200):
#             p.append(sum([w[j] * X[i][j] for j in range(len(trainingSet.columns))]))
        
#         print predictions, p 
        #print len(predictions)
        #print np.dot(w, X[0]), predictions[0]     
        error = 0
        gradient = np.zeros(len(w))
        for i in range(len(predictions)):
            if predictions[i] * Y[i] < 1.0:
                error += 1
                #gradient -= 1.0 * Y[i] * X[i]
                gradient -= np.multiply(X[i], Y[i])
            
        gradient /= 1.0 * len(train_labels)
        #print gradient.shape, X[0].shape
        
        for j in range(1, len(gradient)):
            gradient[j] += 1.0 * regularization * w[j]

        w -= 1.0 * step_size * gradient
        norm_new = np.linalg.norm(w)
        diff = abs(norm_new - norm_old)
        #print count, diff, error
    #print w
    return w


# In[5]:


def get_accuracy_lr(w, trainingSet, testSet):
    total_train = len(trainingSet)
    count_train = 0
    total_test = len(testSet)
    count_test = 0
    
    train_labels = trainingSet['decision']
    test_labels = testSet['decision']
    
    trainingSet = trainingSet.drop('decision', axis=1)
    testSet = testSet.drop('decision', axis=1)
    
    # Training accuracy
    X = np.array(trainingSet)
    Y = np.array(train_labels)
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((X, intercept))
    
    scores = np.dot(X, w)
    predictions = sigmoid(scores)
        
    for i in range(len(predictions)):
        if predictions[i] > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    for i in range(len(predictions)):    
        if predictions[i] == int(train_labels[i]):
            count_train += 1

    training_accuracy = 1.0 * count_train/total_train
    print 'Training Accuracy LR:', '%.2f' % training_accuracy
    
    # Test accuracy
    X = np.array(testSet)
    Y = np.array(test_labels)
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((X, intercept))

    scores = np.dot(X, w)
    predictions = sigmoid(scores)

    for i in range(len(predictions)):
        if predictions[i] > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    for i in range(len(predictions)):    
        if predictions[i] == int(test_labels[i]):
            count_test += 1
            
    test_accuracy = 1.0 * count_test/total_test
    print 'Test Accuracy LR:', '%.2f' % test_accuracy
    
def get_accuracy_svm(w, trainingSet, testSet):
    total_train = len(trainingSet)
    count_train = 0
    total_test = len(testSet)
    count_test = 0
    
    train_labels = trainingSet['decision']
    test_labels = testSet['decision']
    
    trainingSet = trainingSet.drop('decision', axis=1)
    testSet = testSet.drop('decision', axis=1)
    
    # Training accuracy
    X = np.array(trainingSet)
    Y = np.array(train_labels)
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((intercept, X))
    
    predictions = np.dot(X, w)
    
    for i in range(len(predictions)):
        if predictions[i] > 0.0:
            predictions[i] = 1
        else:
            predictions[i] = 0

    for i in range(len(predictions)):    
        if predictions[i] == int(Y[i]):
            count_train += 1

    training_accuracy = 1.0 * count_train/total_train
    print 'Training Accuracy SVM:', '%.2f' % training_accuracy
    
    # Test accuracy
    X = np.array(testSet)
    Y = np.array(test_labels)
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((intercept, X))

    predictions = np.dot(X, w)

    for i in range(len(predictions)):
        if predictions[i] > 0.0:
            predictions[i] = 1
        else:
            predictions[i] = 0

    for i in range(len(predictions)):    
        if predictions[i] == int(Y[i]):
            count_test += 1
            
    test_accuracy = 1.0 * count_test/total_test
    print 'Test Accuracy SVM:', '%.2f' % test_accuracy


# In[6]:


trainingSet = pd.read_csv(trainingDataFilename)
testSet = pd.read_csv(testDataFilename)

if modelIdx == 1:
    w = lr(trainingSet, testSet)
    get_accuracy_lr(w, trainingSet, testSet)
else:
    w = svm(trainingSet, testSet)
    get_accuracy_svm(w, trainingSet, testSet)


# In[ ]:





# In[ ]:




