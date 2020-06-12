#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:29:25 2020

@author: legendary_yin
"""

#doc2vec demo
import gensim
import os
import collections
import smart_open
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D

def splittrain_test(abstract, stopwords_string,test_rate = 0.2):
    train = []
    test = []
    train_tf = []
    test_tf = []
    #random sorted
    #abstract = abstract.iloc[np.random.permutation(len(abstract))]
    
    for i in range(len(abstract)):
        if i <= len(abstract) - np.floor(test_rate * len(abstract)):
           # For training data, add tags
            op_temp = re.sub('[0-9]*',"",str(abstract.iloc[i]))
            op = re.sub(stopwords_string,"",op_temp.lower())
            temp = gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(op), [i]) 
            train.append(temp)
            train_tf.append(op)
        else:
            op_temp = re.sub('[0-9]*',"",str(abstract.iloc[i]))
            op = re.sub(stopwords_string,"",op_temp.lower())
            temp = gensim.utils.simple_preprocess(op)
            test.append(temp)
            test_tf.append(op)
    
    return train, test, train_tf, test_tf

#data = pd.read_csv("D:/BERKELEY-GRADUATE/Virtual reality as a technology strategy tool_case studies in the autonomous driving industry/spring semester/test-sens.csv")
#abstract = data.Abstract

############ use data claim####################################################
data = pd.read_csv("D:/BERKELEY-GRADUATE/E295/round4/expansion_4_1.csv")
#data = data.dropna(axis=0, how='any')
abstract = data.Abstract
title = data.Title
#abstract_title = pd.Series()
#for i in range(len(title)):
    #abstract_title[str(i)] = title[i] + abstract[i]

tf = abstract

###############################################################################
#remove dominant words
##td-idf#######################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(tf)
dense_X = X.todense()
idf = vectorizer.idf_
featurename1 = vectorizer.get_feature_names()
#print(dict(zip(vectorizer.get_feature_names(), idf)))

#get dominant words
one = dense_X > 0
frequency1 = sum(one)
#plt.plot(np.transpose(frequency1))
#By looking at the frequency of each word, find the threshold
#400, frequency > 500 are dominant words
do = pd.Series(frequency1.getA()[0],index = featurename1)
freq_sort1 = do.sort_values(ascending=False)
c1 = freq_sort1[:20].index


index = np.where(frequency1 > 1000)[1]
stopwords = [featurename1[x] for x in index]
stopwords

#for abstract#########
#stopwords_string = r'based([a-z]*)|invent([a-z]*)|time|communication|configured|disclosed|speed|signal|mount([a-z]*)|driv([a-z]*)|motion([a-z]*)|receiv([a-z]*)|associat([a-z]*)|motor|determin([a-z]*)|road|assembly|predetermin([a-z]*)|position([a-z]*)|apparatus([a-z]*)|second([a-z]*)|motion([a-z]*)|sensor([a-z]*)|unit([a-z]*)|plurality([a-z]*)|control([a-z]*)|use([a-z]*)|data([a-z]*)|device([a-z]*)|hav([a-z]*)|includ([a-z]*)|information([a-z]*)|method([a-z]*)|operat([a-z]*)|provid([a-z]*)|vehicle([a-z]*)|systems([a-z]*)|conditions([a-z]*)|autonomous([a-z]*)|using([a-z]*)|drive([a-z]*)|process([a-z]*)|wheel([a-z]*)'
stopwords_string = r'based([a-z]*)|invent([a-z]*)|communication|configured|disclosed|signal|mount([a-z]*)|driv([a-z]*)|motion([a-z]*)|receiv([a-z]*)|associat([a-z]*)|motor|determin([a-z]*)|road|assembly|predetermin([a-z]*)|position([a-z]*)|apparatus([a-z]*)|second([a-z]*)|sensor([a-z]*)|unit([a-z]*)|plurality([a-z]*)|control([a-z]*)|use([a-z]*)|data([a-z]*)|device([a-z]*)|hav([a-z]*)|includ([a-z]*)|information([a-z]*)|method([a-z]*)|operat([a-z]*)|provid([a-z]*)|vehicle([a-z]*)|systems([a-z]*)|conditions([a-z]*)|autonomous([a-z]*)|using([a-z]*)|drive([a-z]*)|process([a-z]*)|wheel([a-z]*)'

#stopwords_string = r''
#for title
#stopwords_string = r'avoidance|collision|control([a-z]*)|motion([a-z]*)|motor([a-z]*)|unmann([a-z]*)|remote([a-z]*)|systems([a-z]*)|auto([a-z]*)|using([a-z]*)|apparatus([a-z]*)|based([a-z]*)|data([a-z]*)|device([a-z]*)|method([a-z]*)|vehicle([a-z]*)|engine|wheel|information|driv([a-z]*)|speed'

###############################################################################
#prepare for doc2vec
testrate = 0
# for doc2vec
train,test, train_tf,test_tf = splittrain_test(tf,stopwords_string, test_rate = testrate)

################
#train the doc2vec model
vector_size = 3
window_size = 10
word_min_count = 1

#model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=1, iter=20)
#model = gensim.models.doc2vec.Doc2Vec(min_count=word_min_count, size=vector_size, alpha=0.025, min_alpha=0.025)
model = gensim.models.doc2vec.Doc2Vec(dm=1,min_count=word_min_count, size=vector_size, iter = 50)

model.build_vocab(train)
model.train(train, total_examples=model.corpus_count, epochs=model.iter)

'''
################
#assessing the model
ranks = []
second_ranks = []
for doc_id in range(len(train)):
    inferred_vector = model.infer_vector(train[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    #find self-similarity rank
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])
    
################
#testing the model
#Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test) - 1)
inferred_vector = model.infer_vector(test[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train[sims[index][0]].words)))
'''

matrix = np.array(model.docvecs[0])
for i in range(len(model.docvecs)-1):
    matrix = np.vstack((matrix,model.docvecs[i+1]))
    
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(matrix)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  
#plt.plot(pca.singular_values_)
norm_singularvalue = pca.singular_values_/sum(pca.singular_values_)
plt.figure()
plt.plot(norm_singularvalue,marker='*')

#####project the original data
project = pca.transform(matrix)

cluster_data = project[:,:]
#cluster_data = cluster_data.reshape(-1,1)
###############################################################################
from sklearn.cluster import KMeans
numcluster = 5
kmeans = KMeans(n_clusters=numcluster, random_state=0,max_iter=1500).fit(cluster_data)
kmeans.labels_

color = ['r','b','y','g','m','c']
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for i in range(numcluster):
    temp_plot = project[np.where(kmeans.labels_ == i),:][0]
    ax.scatter(temp_plot[:,0],temp_plot[:,1],temp_plot[:,2],c=color[i],label='Cluster' + str(i))

ax.legend()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

###############################################################################
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(train_tf)
dense_X = X.todense()
idf = vectorizer.idf_
featurename = vectorizer.get_feature_names()

#feature interprete
cluster = []
cluster_freq = []
cluster_words = []


for i in range(numcluster):
    index = np.where(kmeans.labels_==i)
    each = dense_X[index,:][0]
    cluster.append(each)
    
    each_new = each>0
    frequency = sum(each_new,1)
    cluster_freq.append(frequency) 
    
#threshold = 100

for i in range(numcluster):
    #plt.figure()
    #plt.plot(np.transpose(cluster_freq[i]))
    #index = np.where(np.array(cluster_freq[i])[0]>threshold)
    #c = [featurename[i] for i in index[0]]
    
    freq = cluster_freq[i]
    freq_array = freq.getA()[0]
    bSer= pd.Series(freq_array, index = featurename)
    freq_sort = bSer.sort_values(ascending=False)
    c = freq_sort[:10].index
    cluster_words.append(c)
    
#a = set(cluster_words[0])
#for i in range(len(cluster_words)-1):
   #temp = a & set(cluster_words[i+1])

#word_minus = []
#for i in range(len(cluster_words)):
    #b = set(cluster_words[i]).difference(temp)
    #word_minus.append(b)
#word_minus

'''
def intradistance(a):
    #a is a array type matrix
    if len(a) == 1:
        return 0
    dist_sum = sum([np.sqrt(sum((a[x]-a[y])*(a[x]-a[y]))) for x in range(len(a)) for y in range(x,len(a))])
    dist = dist_sum * 2 / (len(a) * (len(a)-1))
    return dist
def interdistance(a,b):
    #a and b are array type matrix
    dist_sum = sum((sum(a)/len(a)-sum(b)/len(b))*(sum(a)/len(a)-sum(b)/len(b)))
    dist = np.sqrt(dist_sum)
    return dist
#find the best cluster number
from sklearn.cluster import FeatureAgglomeration
intra = []
inter = []
for numcluster in range(5,10):
    #h_clustering = FeatureAgglomeration(n_clusters=numcluster).fit(cluster_data)
    #label = h_clustering.labels_
    kmeans = KMeans(n_clusters=numcluster, random_state=0,max_iter=1000).fit(cluster_data)
    label = kmeans.labels_
    
    cluster = []
    intrasum = 0
    intersum = 0
    for i in range(numcluster):
        index = np.where(label==i)
        each = cluster_data[index,:][0]
        cluster.append(each)
        
        intrasum = intrasum + intradistance(each)
    
    intra.append(intrasum / numcluster)
    intersum = sum([interdistance(cluster[i],cluster[j]) for i in range(numcluster) for j in range(i,numcluster)])
    intersum = intersum * 2 / (numcluster*numcluster-numcluster)
    inter.append(intersum)
    print(numcluster)
    
plt.figure()
plt.plot(intra)
plt.plot(inter)  
plt.show()
'''
'''
# find the meaning the each axis
x = cluster_data[:,0]
xSer= pd.Series(x, index = range(len(x)))
freq_sort = xSer.sort_values(ascending=False)
x_pos = freq_sort[:10].index
x_neg = freq_sort[len(x)-10:].index
                 
y = cluster_data[:,1]
ySer= pd.Series(y, index = range(len(y)))
freq_sort = ySer.sort_values(ascending=False)
y_pos = freq_sort[:10].index
y_neg = freq_sort[len(y)-10:].index
                  
z = cluster_data[:,0]
zSer= pd.Series(z, index = range(len(z)))
freq_sort = zSer.sort_values(ascending=False)
z_pos = freq_sort[:10].index
z_neg = freq_sort[len(z)-10:].index
                  
tf_x_pos = [train_tf[x_pos[w]] for w in range(len(x_pos))]
tf_x_neg = [train_tf[x_neg[w]] for w in range(len(x_neg))]
tf_y_pos = [train_tf[y_pos[w]] for w in range(len(y_pos))]
tf_y_neg = [train_tf[y_neg[w]] for w in range(len(y_neg))]
tf_z_pos = [train_tf[z_pos[w]] for w in range(len(z_pos))]
tf_z_neg = [train_tf[z_neg[w]] for w in range(len(z_neg))]
tf_x_pos = [w.lower().split() for w in tf_x_pos]
tf_x_neg = [w.lower().split() for w in tf_x_neg]
tf_y_pos = [w.lower().split() for w in tf_y_pos]
tf_y_neg = [w.lower().split() for w in tf_y_neg]
tf_z_pos = [w.lower().split() for w in tf_z_pos]
tf_z_neg = [w.lower().split() for w in tf_z_neg]
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
eng_stopwords = set(stopwords.words("english"))
wnl = WordNetLemmatizer()
def clean_patent(most):
    clean=[]
    for word in most:
        if word not in eng_stopwords:
            word=wnl.lemmatize(word)
            clean.append(word)
    return(clean)
tf_x_pos = [clean_patent(w) for w in tf_x_pos]
tf_x_neg = [clean_patent(w) for w in tf_x_neg]
tf_y_pos = [clean_patent(w) for w in tf_y_pos]
tf_y_neg = [clean_patent(w) for w in tf_y_neg]
tf_z_pos = [clean_patent(w) for w in tf_z_pos]
tf_z_neg = [clean_patent(w) for w in tf_z_neg]
from gensim.models import word2vec
model = word2vec.Word2Vec(tf_x_pos,size=100, window=5, min_count=1)
model.wv.vocab
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform([train_tf[x_pos[w]] for w in range(len(x_pos))])
dense_X = X.todense()
idf = vectorizer.idf_
featurename = vectorizer.get_feature_names()
bSer= pd.Series(idf, index = featurename)
freq_sort = bSer.sort_values(ascending=False)
c = freq_sort[:10].index
'''                 
'''
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#find self-similarity rank
rank = [docid for docid, sim in sims].index(doc_id)
ranks.append(rank)
second_ranks.append(sims[1])
model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
'''



from nltk.stem import PorterStemmer
ps = PorterStemmer()
words = []
for i in range(len(cluster_words)):
    temp = []
    for word in cluster_words[i]:
        word=ps.stem(word)
        temp.append(word)
    words.append(temp)
    
a0 = set(words[0])
a1 = set(words[1])
a2 = set(words[2])
a3 = set(words[3])
a4 = set(words[4])
common = a0 | a1 | a2 | a3 | a4

freq_final = pd.DataFrame()
for i in common:
    temp = [i in words[j] for j in range(len(words))]
    freq_final[i] = temp
        
sum_final = freq_final.loc[0] * 1 + freq_final.loc[1] * 1 + freq_final.loc[2] * 1 + freq_final.loc[3] * 1 + freq_final.loc[4] * 1

descriptive = set(sum_final[sum_final >= 3].index)
a0_sep = a0 - descriptive
a1_sep = a1 - descriptive
a2_sep = a2 - descriptive
a3_sep = a3 - descriptive
a4_sep = a4 - descriptive


color = ['r','b','y','g','m']
label = [['TRACKING: track,axis,element,optical,state,radar,unman,automat,arrange,light,connect,image'],
         ['DECISION: acceler,beam,direct,distanc,evaluate,measure,module,object,reflect,transmit,zone,speed,respons'],
         ['CONTROL: combust,engine,mechanism,throttle,valve,air,location'],
         ['MANIPULATION: manipulation,approach,enable,identify,input,intuit,simultan,slide,classif,multiple,connect'],
         ['HARDWARE: amplify,array,circuit,current,element,level,semiconductor,voltag,switch']]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
for i in range(numcluster):
    temp_plot = project[np.where(kmeans.labels_ == i),:][0]
    ax.scatter(temp_plot[:,0],temp_plot[:,1],temp_plot[:,2],c=color[i],label=label[i])

ax.legend()
ax.set_title('Descriptive words: sensor,detect,light,image,response,time,connect,output',)
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
plt.show()