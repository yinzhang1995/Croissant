#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:37:51 2020

@author: legendary_yin
"""

# random test

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats

def generatepopulation(realpdf,s):
    if realpdf == 'uniform':
        return np.random.uniform(0,1,s)
    elif realpdf == 'poisson':
        return np.random.poisson(3,s)
    elif realpdf == 'exponential':
        return np.random.exponential(3,s)
    elif realpdf == 'binomial':
        return np.random.binomial(n = 1, size = s, p = 0.2)
    else:
        return np.random.normal(0,1,s)

def plotpopulation(population):
    plt.figure() 
    plt.hist(population)
    plt.show()

   
def cltproof(population, samplesize, trial):
    if samplesize >= len(population):
        return None
    else:
        sample = []
        for i in range(trial):
            sample.append(random.choices(population,k = samplesize))
    sample = np.matrix(sample)
    samp_mean = sample.mean(1)
    samp_std = np.std(samp_mean)  
    
    plt.figure() 
    plt.hist(samp_mean, density = True)
    x = np.linspace(np.min(samp_mean), np.max(samp_mean), 1000)
    y = 1 / (samp_std * (2 * np.pi) ** 0.5) * np.exp(- np.multiply(x - np.mean(samp_mean),x - np.mean(samp_mean)) / (2 * samp_std * samp_std))
    plt.plot(x,y)
    plt.show()
    
    
def getsample(population, samplesize):
    if samplesize >= len(population):
        return None
    else:
        return random.choices(population,k = samplesize)
    
    
def getci(sample, tail, a = 0.05):
    if tail == 'double':
        moe = scipy.stats.norm.ppf(1 - a/2,0,1) * np.std(sample) / (len(sample) ** 0.5)
        return [np.mean(sample) - moe, np.mean(sample) + moe]
    elif tail == 'left':  # u >= u0
        moe = scipy.stats.norm.ppf(1 - a,0,1) * np.std(sample) / (len(sample) ** 0.5)
        return [np.mean(sample) - moe, 1000]
    else:   # u <= u0
        moe = scipy.stats.norm.ppf(1 - a,0,1) * np.std(sample) / (len(sample) ** 0.5)
        return [-1000, np.mean(sample) + moe]
        
    
def getp(sample,u0, tail):
    if tail == 'double':
        if np.mean(sample) > u0:
            return (1 - scipy.stats.norm.cdf((np.mean(sample) - u0) / (np.std(sample) / (len(sample) ** 0.5)),0,1)) * 2
        else:
            return scipy.stats.norm.cdf((np.mean(sample) - u0) / (np.std(sample) / (len(sample) ** 0.5)),0,1) * 2
    
    elif tail == 'left':  # u >= u0
        
        return scipy.stats.norm.cdf((np.mean(sample) - u0) / (np.std(sample) / (len(sample) ** 0.5)),0,1)
    else:   # u <= u0
        return 1 - scipy.stats.norm.cdf((np.mean(sample) - u0) / (np.std(sample) / (len(sample) ** 0.5)),0,1)
    
population_new = generatepopulation(realpdf = 'uniform',s = 10000)
#plotpopulation(population_new)
#cltproof(population_new, samplesize = 100, trial = 100)

y = []
dy = []
ci_all = []
p_all = []
trial = 100
for i in range(trial):
 
    sample = getsample(population_new,1000)
    ci = getci(sample, tail = 'double',a = 0.05)
    p =  getp(sample,0.5,tail = 'double')
    y.append(np.mean(ci))
    dy.append((ci[1] - ci[0]) / 2)
    ci_all.append(ci)
    p_all.append(p)

x = np.arange(trial)

a = (np.array(ci_all)[:,0] > 0.5) | (np.array(ci_all)[:,1] < 0.5)
c = ['red' if i else 'black' for i in a]
plt.figure()
plt.subplot(121)
plt.errorbar(x, y, yerr=dy, fmt='.k', ecolor = c)
plt.plot(x,[np.mean(population_new)] * len(x))
plt.subplot(122)
plt.plot(x,p_all)


#######
# AB Test start
population_new = generatepopulation(realpdf = 'uniform',s = 1000000)

y = []
dy = []
ci_all = []
p_all = []
test_sample_mean = []
compare_sample_mean = []
trial = 100
sample_size = 1000
for i in range(trial):
 
    compare_group = np.array(getsample(population_new,sample_size))
    test_group = np.array(getsample(population_new,sample_size)) + 0.02   # test makes the mean increases 4%
    compare_sample_mean.append(np.mean(compare_group))
    test_sample_mean.append(np.mean(test_group))
    
    ci = getci(test_group - compare_group, tail = 'double',a = 0.05)
    p =  getp(test_group - compare_group,0.0,tail = 'double')
    
    y.append(np.mean(ci))
    dy.append((ci[1] - ci[0]) / 2)
    ci_all.append(ci)
    p_all.append(p)

x = np.arange(trial)

a = (np.array(ci_all)[:,0] > 0.0) | (np.array(ci_all)[:,1] < 0.0)
c = ['red' if i else 'black' for i in a]
plt.figure()
plt.subplot(221)
plt.errorbar(x, y, yerr=dy, fmt='.k', ecolor = c)
plt.plot(x,[0] * len(x))
plt.subplot(222)
plt.plot(x,p_all,'.--')
plt.plot(x,[0.05] * len(x))
plt.subplot(223)
plt.hist(test_sample_mean, alpha = 0.5)
plt.hist(compare_sample_mean, alpha = 0.5)
# beta is so high because the sample size is small

std_pop = np.std(population_new)



# standard error of test group is different from standard error of compared group
population_new = generatepopulation(realpdf = 'binomial',s = 1000000)

y = []
dy = []
ci_all = []
p_all = []
test_sample_mean = []
compare_sample_mean = []
trial = 100
sample_size = 1540
for i in range(trial):
 
    compare_group = np.array(getsample(population_new,sample_size))
    test_group = np.array(getsample(population_new,sample_size)) 
    
    # change some test samples to 1
    index = np.arange(0,len(test_group),1)
    temp1 = index[test_group == 0]
    temp2 = random.choices(temp1, k = int(len(temp1) * 0.05))
    test_group[temp2] = 1
    
    compare_sample_mean.append(np.mean(compare_group))
    test_sample_mean.append(np.mean(test_group))
    
    ci = getci(test_group - compare_group, tail = 'double',a = 0.05)
    p =  getp(test_group - compare_group,0.0,tail = 'double')
    
    y.append(np.mean(ci))
    dy.append((ci[1] - ci[0]) / 2)
    ci_all.append(ci)
    p_all.append(p)

x = np.arange(trial)

a = (np.array(ci_all)[:,0] > 0.0) | (np.array(ci_all)[:,1] < 0.0)
c = ['red' if i else 'black' for i in a]
plt.figure()
plt.subplot(221)
plt.errorbar(x, y, yerr=dy, fmt='.k', ecolor = c)
plt.plot(x,[0] * len(x))
plt.subplot(222)
plt.plot(x,p_all,'.--')
plt.plot(x,[0.05] * len(x))
plt.subplot(223)
plt.hist(test_sample_mean, alpha = 0.5)
plt.hist(compare_sample_mean, alpha = 0.5)


std_pop = np.std(population_new)



np.std(compare_group)
np.std(test_group)

# how many sample do I need?
#population_new = generatepopulation(realpdf = 'binomial',s = 1000000)
samp = np.array(getsample(population_new,2000))
std_est = np.std(samp)

prob = sum(samp) / len(samp)
std_est2 = (prob * (1 - prob)) ** 0.5


sample_sz = 1000
practical_diff = 0.03
a = 0.05
b_target = 0.1

quantile = scipy.stats.norm.ppf(1 - a, 0, std_est / (sample_sz ** 0.5))
b = scipy.stats.norm.cdf(quantile,practical_diff,std_est / (sample_sz ** 0.5))
while b >= b_target:
    sample_sz = sample_sz + 10
    quantile = scipy.stats.norm.ppf(1 - a, 0, std_est / (sample_sz ** 0.5))
    b = scipy.stats.norm.cdf(quantile,practical_diff,std_est / (sample_sz ** 0.5))









    