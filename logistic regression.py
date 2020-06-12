#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:09:04 2019

@author: legendary_yin
"""

##############################
# Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

samplesize = 100

Xtemp = {'A':np.random.randint(-100,100,size = samplesize), 'B':np.random.randint(-50,50,size = samplesize), 'C':np.array([1] * samplesize)}
X = pd.DataFrame(Xtemp)
Ytemp = 2 * X['A'] + 5 * X['B'] + X['C']
Ytemp3 = Ytemp + np.random.normal(loc = 0, scale = 1, size = samplesize)
Ytemp1 = np.exp(Ytemp3) / (1 + np.exp(Ytemp3))

Y = Ytemp1
Y[Ytemp1 >= 0.5] = 1
Y[Ytemp1 < 0.5] = 0

X.describe()
X.hist()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(X['A']), np.array(X['B']), np.array(Y), marker = 'o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


# grediant decent
alpha = 0.0001
theta = [2,2,2]
loss_step = []
for s in range(100):
    loss = 0
    for i in range(samplesize):
        loss = loss - (Y[i] * (theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) - np.log(1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i])))
    loss_step.append(loss)
    temp0 = 0
    temp1 = 0
    temp2 = 0
    for i in range(samplesize):
        temp0 = temp0 + Y[i] * X['A'][i] - X['A'][i] * np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) / (1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]))
        temp1 = temp1 + Y[i] * X['B'][i] - X['B'][i] * np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) / (1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]))
        temp2 = temp2 + Y[i] * X['C'][i] - X['C'][i] * np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) / (1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]))

    theta[0] = theta[0] + alpha * temp0
    theta[1] = theta[1] + alpha * temp1
    theta[2] = theta[2] + alpha * temp2
    
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.arange(100),np.array(loss_step))

# stochastic gradiant decent

alpha = 0.0001
theta = [2,2,2]
loss_step = []
for s in range(100):
    loss = 0
    for i in range(samplesize):
        loss = loss - (Y[i] * (theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) - np.log(1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i])))
    loss_step.append(loss)
    temp0 = 0
    temp1 = 0
    temp2 = 0
    subsample = np.random.choice(100, 20, replace=False)
    for i in subsample:
        temp0 = temp0 + Y[i] * X['A'][i] - X['A'][i] * np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) / (1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]))
        temp1 = temp1 + Y[i] * X['B'][i] - X['B'][i] * np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) / (1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]))
        temp2 = temp2 + Y[i] * X['C'][i] - X['C'][i] * np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]) / (1 + np.exp(theta[0] * X['A'][i] + theta[1] * X['B'][i] + theta[2] * X['C'][i]))

    theta[0] = theta[0] + alpha * temp0
    theta[1] = theta[1] + alpha * temp1
    theta[2] = theta[2] + alpha * temp2
    
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.arange(100),np.array(loss_step))













samplesize = 100

Xtemp = {'A':np.random.randint(-100,100,size = samplesize),'C':np.array([1] * samplesize)}
X = pd.DataFrame(Xtemp)
Ytemp = 2 * X['A'] + X['C']
Ytemp3 = Ytemp + np.random.normal(loc = 0, scale = 5, size = samplesize)
Ytemp1 = np.exp(Ytemp3) / (1 + np.exp(Ytemp3))

Y = Ytemp1
Y[Ytemp1 >= 0.5] = 1
Y[Ytemp1 < 0.5] = 0


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.array(X['A']),np.array(Y), marker = 'o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Y.value_counts()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=100)
print ('Number of samples in training data:',len(x_train))
print ('Number of samples in validation data:',len(x_test))

logreg_model = LogisticRegression(C=1000)
print ('Training a logistic Regression Model..')
logreg_model.fit(x_train, y_train)
training_accuracy1=logreg_model.score(x_train,y_train)   #Returns the mean accuracy on the given test data and labels
print ('Training Accuracy:',training_accuracy1)

validation_accuracy1=logreg_model.score(x_test,y_test)
print('Accuraacy of the model on test data: ',validation_accuracy1)





from sklearn.metrics import confusion_matrix   
#By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
print ('Calculating the confusion matrix for training and test data...')
y_true = y_train
y_pred = logreg_model.predict(x_train)
cftrain1=pd.DataFrame(confusion_matrix(y_true, y_pred))#,columns=['Pred 0',1,2],index=['Act 0',1,2])
print ('Confusion matrix of training data is: \n',cftrain1)

y_true = y_test
y_pred = logreg_model.predict(x_test)
cftest1=pd.DataFrame(confusion_matrix(y_true, y_pred),columns=['Pred 0',1,2],index=['Act 0',1,2])
print ('Confusion matrix of test data is: \n',cftest1)



y_score1 = logreg_model.predict_proba(x_test)[:,1]
from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)

