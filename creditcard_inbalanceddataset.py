#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:51:22 2020

@author: legendary_yin
"""

# credit fraud
# imbalanced data

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

os.getcwd()
os.chdir('/Users/legendary_yin/Desktop/data scientist review/training/')

data = pd.read_csv('creditcard.csv')


data.Class.value_counts()
data.Class.plot(kind = 'hist')
data.isnull().sum()


Y = data['Class']
data.pop('Class')


# KMeans
kmeans = KMeans(n_clusters = 2, random_state = 0).fit(data)
kmeans.labels_

confusion_matrix( Y,kmeans.labels_)

logmodel = LogisticRegression(penalty = 'l1')
logmodel.fit(data,Y)

y_pred = logmodel.predict_proba(data)

logmodel.coef_

fpr,tpr,threshold = roc_curve(Y, y_pred[:,[1]])

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

precision,recall,threshold = precision_recall_curve(Y, y_pred[:,[1]])
plt.figure()
plt.plot(precision, recall)
plt.xlabel('precision')
plt.ylabel('recall')
plt.show()

threshold[(precision[:-1] >= 0.75) & (precision[:-1] <= 0.8) & (recall[:-1] >= 0.75) & (recall[:-1] <= 0.8)]


y_pred1 = y_pred[:,[1]]
y_pred1[y_pred[:,[1]] >= 0.1] = 1
y_pred1[y_pred[:,[1]] < 0.1] = 0

confusion_matrix(Y, y_pred1)


#### train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size = 0.3, random_state = 1)

Y_train.value_counts()
Y_test.value_counts()

logmodel = LogisticRegression(penalty = 'l1', C = 0.1)
logmodel.fit(X_train,Y_train)

y_pred = logmodel.predict_proba(X_train)

logmodel.coef_

fpr,tpr,threshold = roc_curve(Y_train, y_pred[:,[1]])

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

precision,recall,threshold = precision_recall_curve(Y_train, y_pred[:,[1]])
plt.figure()
plt.plot(precision, recall)
plt.xlabel('precision')
plt.ylabel('recall')
plt.show()

threshold[(precision[:-1] >= 0.8) & (precision[:-1] <= 0.85) & (recall[:-1] >= 0.8) & (recall[:-1] <= 0.85)]


y_pred1 = y_pred[:,[1]]
y_pred1[y_pred[:,[1]] >= 0.09] = 1
y_pred1[y_pred[:,[1]] < 0.09] = 0

confusion_matrix(Y_train, y_pred1)

#array([[198938,     69],
#       [    69,    288]])



# repeat till having the same scale
X_train1 = X_train.loc[Y_train == 1, :]
X_train0 = X_train.loc[Y_train == 0, :]

X_trainnew = X_train0
Y_trainnew = Y_train[Y_train == 0]
from random import randint
from random import choices
import numpy as np 

for i in range(190):
    temp = choices(np.arange(X_train1.shape[0]), k = 1000)
    X_trainnew = X_trainnew.append(X_train1.iloc[temp,:])
    Y_trainnew = Y_trainnew.append(pd.Series([1] * 1000))
    print(i)
    
logmodel = LogisticRegression(penalty = 'l1', C = 0.1)
logmodel.fit(X_trainnew,Y_trainnew)

y_pred = logmodel.predict_proba(X_trainnew)

logmodel.coef_

fpr,tpr,threshold = roc_curve(Y_trainnew, y_pred[:,[1]])

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

b = threshold[(fpr >= 0.1) & (fpr <= 0.12) & (tpr >= 0.96) & (tpr <= 0.97)]

# precision recall curve for balanced data is not useful
#precision,recall,threshold = precision_recall_curve(Y_trainnew, y_pred[:,[1]])
#plt.figure()
#plt.plot(recall,precision)
#plt.xlabel('recall')
#plt.ylabel('precision')
#plt.show()

#a = threshold[(precision[:-1] >= 0.98) & (precision[:-1] <= 0.99) & (recall[:-1] >= 0.92) & (recall[:-1] <= 0.93)]


y_pred1 = y_pred[:,[1]]
y_pred1[y_pred[:,[1]] >= 0.16] = 1
y_pred1[y_pred[:,[1]] < 0.16] = 0

confusion_matrix(Y_trainnew, y_pred1)

#array([[174872,  24135],
#       [  6948, 183052]])

plt.subplot(5,6,1)
for i in range(1,31):
    plt.subplot(5,6,i)
    plt.boxplot(pd.concat([X_trainnew.iloc[i,:], Y_trainnew], axis=1))



