#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:27:22 2019

@author: legendary_yin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Linear Regression 
samplesize = 100

Xtemp = {'A': np.random.randint(0,100,size = samplesize), 'B': np.random.randint(0,50,size = samplesize), 'C': np.array([1] * samplesize)}
X = pd.DataFrame(Xtemp)
Y = X['A'] * 2 + X['B'] * (-3) + 5 * X['C']

# Add residuals
Y1 = Y + np.random.normal(loc = 0, scale = 20, size = samplesize)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(X['A']), np.array(X['B']), np.array(Y1), marker = 'o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.array(X['A']), np.array(X['B']), marker = 'o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()


# Normal Equation
theta = np.linalg.inv(np.transpose(X.values).dot(X.values)).dot(np.transpose(X.values)).dot(Y1)

# Gradient Descent
alpha = 0.000001
theta_start = np.array([0,0,0])
cost = []
for i in range(100):
    theta_start = theta_start - 2 * alpha * (np.transpose(X.values).dot(X.values).dot(theta_start) - np.transpose(X.values).dot(Y1.values))
    cost_function = np.transpose(Y1 - X.values.dot(theta_start)).dot(Y1 - X.values.dot(theta_start))
    cost.append(cost_function)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range(100), cost, marker = 'o')
ax.set_xlabel('Step')
ax.set_ylabel('Cost Function')
plt.show()


# Use Package
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True,normalize = True).fit(X[['A','B']], Y1)
model.intercept_
model.coef_







