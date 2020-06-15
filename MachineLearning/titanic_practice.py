#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:44:31 2020

@author: legendary_yin
"""

# Titanic Practice
'''
pclass: Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival: A Boolean indicating whether the passenger survived or not (0 = No; 1 = Yes); this is our target
name: A field rich in information as it contains title and family names
sex: male/female
age: Age, asignificant portion of values aremissing
sibsp: Number of siblings/spouses aboard
parch: Number of parents/children aboard
ticket: Ticket number.
fare: Passenger fare (British Pound).
cabin: Doesthe location of the cabin influence chances of survival?
embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat: Lifeboat, many missing values
body: Body Identification Number
home.dest: Home/destination
'''


import pandas as pd
import numpy as np

data = pd.read_csv('/Users/legendary_yin/Desktop/data scientist review/Amazon/titanic.csv')

# distribution of int/float variables
data.hist()

# type of each column
data.dtypes

# missing values
data.isnull().sum()

data.describe()

pd.scatter_matrix(data,figsize = (16,16))

data.isnull().sum()


# fill in age null values
new = data.name.str.split(pat='.',n=1,expand = True)
data['lastname'] = new[1]
# data['title'] = new[0].str.split(pat=' ',n=1,expand = True)[1]


data['isMaster'] = data.name.str.contains(pat = 'Master')
data['isMrs'] = data.name.str.contains(pat = 'Mrs')
data['isMiss'] = data.name.str.contains(pat = 'Miss')
data['isMr'] = data.name.str.contains(pat = 'Mr')
data['isDr'] = data.name.str.contains(pat = 'Dr')

data['title'] = 'Other'

data.loc[data.isMaster == True,'title'] = 'Master'
data.loc[data.isMrs == True,'title'] = 'Mrs'
data.loc[data.isMiss == True,'title'] = 'Miss'
data.loc[data.isMr == True,'title'] = 'Mr'
data.loc[data.isDr == True,'title'] = 'Dr'

data.drop(columns = ['body','boat','home.dest','cabin','ticket',
                     'isMaster','isMrs','isMiss','isMr','isDr'], axis = 1,inplace = True)

data.loc[(data.age.isnull()) & (data.title == 'Master'), 'age'] = np.mean(data.loc[data.title == 'Master', 'age'])
data.loc[(data.age.isnull()) & (data.title == 'Mrs'), 'age'] = np.mean(data.loc[data.title == 'Mrs', 'age'])
data.loc[(data.age.isnull()) & (data.title == 'Mr'), 'age'] = np.mean(data.loc[data.title == 'Mr', 'age'])
data.loc[(data.age.isnull()) & (data.title == 'Miss'), 'age'] = np.mean(data.loc[data.title == 'Miss', 'age'])
data.loc[(data.age.isnull()) & (data.title == 'Dr'), 'age'] = np.mean(data.loc[data.title == 'Dr', 'age'])
data.loc[(data.age.isnull()) & (data.title == 'Other'), 'age'] = np.mean(data.loc[data.title == 'Other', 'age'])

data.isnull().sum()

data.dropna(inplace = True)

data.drop(columns = ['name','title','lastname'], axis = 1, inplace = True)

# one-hot encoding
nonnum = data.columns[data.dtypes.isin(['object'])]

data_onehot = pd.get_dummies(data,columns = nonnum,drop_first = True)


# train model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix   
from sklearn.metrics import roc_curve,auc,precision_recall_curve

Y = data_onehot['survived']
X = data_onehot[data_onehot.columns[~data_onehot.columns.isin(['survived'])]]
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 100)

logreg_model = LogisticRegression(C=0.1,penalty = 'l1')
print ('Training a logistic Regression Model..')
logreg_model.fit(x_train, y_train)

y_train_pred_prob = logreg_model.predict_proba(x_train)
y_train_pred = logreg_model.predict(x_train)

confusion_matrix(y_train, y_train_pred)

training_accuracy1=logreg_model.score(x_train,y_train)   #Returns the mean accuracy on the given test data and labels
print ('Training Accuracy:',training_accuracy1)

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_train, y_train_pred_prob[:,[1]])

logreg_model.coef_



import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(false_positive_rate1, true_positive_rate1, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc(false_positive_rate1, true_positive_rate1))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

threshold1[(true_positive_rate1 >= 0.77) & (false_positive_rate1 <= 0.21)]
threshold = 0.375

y_train_pred_new = y_train_pred_prob[:,1]
y_train_pred_new[y_train_pred_new >= threshold] = 1
y_train_pred_new[y_train_pred_new < threshold] = 0

confusion_matrix(y_train, y_train_pred_new)


precision1, recall1, threshold1 = precision_recall_curve(y_train, y_train_pred_prob[:,[1]])

plt.figure()
lw = 2
plt.plot(precision1, recall1, color='darkorange',
         lw=lw)
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision Recall example')
#plt.legend(loc="lower right")
plt.show()

y_test_pred_prob = logreg_model.predict_proba(x_test)
y_test_pred_new = y_test_pred_prob[:,1]
y_test_pred_new[y_test_pred_new >= threshold] = 1
y_test_pred_new[y_test_pred_new < threshold] = 0

confusion_matrix(y_test,y_test_pred_new)
logreg_model.score(x_test,y_test_pred_new)

logreg_model.coef_




############
from sklearn.model_selection import KFold
# 5 fold cross validation
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 1.0/6, random_state = 100)

error_final = []
#para_all = np.linspace(0,100,10000)
#para_all = para_all[1:]
para_all = [0.001,0.005, 0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]

roc_fp_x = []
roc_tp_y = []


for i in para_all:
    kf = KFold(n_splits = 5, shuffle = True)
    sumerror = 0
    for train_index, test_index in kf.split(x_train):
        x_train1, x_val1 = x_train.iloc[train_index,], x_train.iloc[test_index,]
        y_train1, y_val1 = y_train.iloc[train_index], y_train.iloc[test_index]
        
        logreg_model = LogisticRegression(C = i, penalty = 'l1')
        
        logreg_model.fit(x_train1, y_train1)
        
        y_train1_pred_prob = logreg_model.predict_proba(x_train1)
        
        # get best threshold
        fp1, tp1, th = roc_curve(y_train1, y_train1_pred_prob[:,1])
        
        roc_fp_x.append(fp1)
        roc_tp_y.append(tp1)
        
        temp = np.absolute(np.array(tp1) / np.array(fp1) - 2)
        bestth = th[temp == np.nanmin(temp)]
        
        y_val1_pred_prob = logreg_model.predict_proba(x_val1)
        y_val1_pred = y_val1_pred_prob[:,1]
        y_val1_pred[y_val1_pred_prob[:,1] >= bestth] = 1
        y_val1_pred[y_val1_pred_prob[:,1] < bestth] = 0
        
        cm = confusion_matrix(y_val1, y_val1_pred)
        error = 1 - sum(cm[k][k] for k in range(2)) * 1.0 / np.sum(cm)
        sumerror = sumerror + error
        
    error_final.append(sumerror / 5)
        
# plot cross validation result
plt.figure()
plt.plot(para_all, error_final,color='darkorange',lw=2)
plt.xlabel('Parameter')
plt.ylabel('Error')
plt.show()


# plot roc  
labels = []
for j in range(len(roc_fp_x)):
    plt.plot(roc_fp_x[j], roc_tp_y[j])
    labels.append('ROC curve (area = %0.2f)' % auc(roc_fp_x[j], roc_tp_y[j]))

# I'm basically just demonstrating several different legend options here...
plt.legend(labels, ncol=4, loc='upper center', 
           bbox_to_anchor=[0.5, 1.1], 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True, shadow=True)

plt.show()
  

###########################################
# random forest

# max_features
# n_estimators
# min_samples_leaf


import matplotlib.pyplot as plt
from sklearn.metrics import KFold
fig,axs = plt.subplots(1,2)
data[data.sex == 'female'].survived.value_counts().plot(kind = 'barh',ax = axs[0],title = 'Female Survival')
data[data.sex == 'male'].survived.value_counts().plot(kind = 'barh',ax = axs[1],title = 'Male Survival')


# train model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix   
from sklearn.metrics import roc_curve,auc,precision_recall_curve

data_onehot = pd.get_dummies(data, columns = data.columns[data.dtypes == 'object'],drop_first = True)

Y = data_onehot.pop('survived')
X = data_onehot
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 1.0/6, random_state = 100)

rf_model = RandomForestClassifier(n_estimators = 100, oob_score = True, n_jobs = 1, min_samples_leaf = 5, max_features = 'sqrt')
print ('Training a logistic Regression Model..')
rf_model.fit(x_train, y_train)

y_train_pred_prob = rf_model.predict_proba(x_train)
y_train_pred = rf_model.predict(x_train)

confusion_matrix(y_train, y_train_pred)

rf_model.oob_score_
rf_model.feature_importances_

fi = pd.Series(rf_model.feature_importances_, index = x_train.columns)
fi.sort_values().plot(kind='barh')



from sklearn.model_selection import KFold
auc_final = []
for i in [10,20,50,100,200,500,800,1000,2000]:
    for j in [1,2,3,4,5,6,7,8,9,10]:
        kf = KFold(n_splits = 5, shuffle = True)
        sumauc = 0
        for train_index, test_index in kf.split(x_train):
            x_train1, x_val1 = x_train.iloc[train_index,], x_train.iloc[test_index,]
            y_train1, y_val1 = y_train.iloc[train_index], y_train.iloc[test_index]
            
            rf_model = RandomForestClassifier(n_jobs = 1, n_estimators = i, max_features = 'sqrt',min_samples_leaf = j)
            
            rf_model.fit(x_train1, y_train1)
            
            y_train1_pred_prob = rf_model.predict_proba(x_train1)
            
            # get best threshold
            fp1, tp1, th = roc_curve(y_train1, y_train1_pred_prob[:,1])
            
            sumauc = sumauc + auc(fp1, tp1)
            
        auc_final.append(sumauc/5)   


plotX,plotY = np.meshgrid([1,2,3,4,5,6,7,8,9,10],[10,20,50,100,200,500,800,1000,2000])

plotZ = np.reshape(auc_final, (9,10))

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(plotX, plotY, plotZ, linewidth=0, antialiased=False)
plt.show()

