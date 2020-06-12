#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:58:33 2020

@author: legendary_yin
"""

import pandas as pd
import numpy as np
import os

pth = os.getcwd()

os.chdir("/Users/legendary_yin/Desktop/data scientist review/training/twitter-sentiment-analysis2/")

pth = os.getcwd()
pth

s = "train.csv"
s

twitter = pd.read_csv(s,encoding='latin-1')
# set the encoding to ‘latin-1’ as the text had many special characters.

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# drop na
twitter.columns
twitter.dropna(subset = ['SentimentText'], inplace = True)

# change to lower case
twitter['lctext'] = [i.lower() for i in twitter.SentimentText]

# tokenization: breaking a stream of text up into words, phrases, symbols, 
#or other meaningful elements called tokens. 

# work_tokenize()   break a stream of text into words
twitter['tokentext'] = [word_tokenize(i) for i in twitter.lctext]
# sent_tokenize()   break a stream of text into sentence
# twitter['senttokentext'] = [sent_tokenize(i) for i in twitter.lctext]

twitter.head(10)
#import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


#Word Stemming/Lemmatization: The aim of both processes is the same, 
#reducing the inflectional forms of each word into a common base or root. 
#Lemmatization is closely related to stemming. The difference is that a stemmer 
#operates on a single word without knowledge of the context, and therefore cannot 
#discriminate between words which have different meanings depending on part of speech. 
#However, stemmers are typically easier to implement and run faster, 
#and the reduced accuracy may not matter for some applications.


for index,entry in enumerate(twitter['tokentext']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    twitter.loc[index,'text_final'] = str(Final_words)


#tf-idf: verctorize features:convert text data to vectors which the model can understand
#“Term Frequency — Inverse Document” Frequency 
#Term Frequency: This summarizes how often a given word appears within a document.
#Inverse Document Frequency: This down scales words that appear a lot across documents.
#TF-IDF are word frequency scores that try to highlight words that are more interesting, e.g. frequent in a document but not across documents.
    
Y = twitter.pop('text_final')
X=Y

Y = twitter.Sentiment

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.3, random_state = 100)
    
Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(X_train)

Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf = Tfidf_vect.transform(X_test)

print(Tfidf_vect.vocabulary_)

print(Train_X_Tfidf)

type(Train_X_Tfidf)  #sparse matrix



# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Y_train)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Y_test))
confusion_matrix(predictions_NB, Y_test)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='poly',degree = 2)
SVM.fit(Train_X_Tfidf,Y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(predictions_SVM, Y_test)




# word2vec
import numpy as np
import gensim 
from gensim.models import Word2Vec 

#The basic idea of word embedding is words that occur in similar context tend to 
#be closer to each other in vector space

model1 = gensim.models.Word2Vec(X_train, min_count = 1, size = 100, window = 5) 
model1.wv.vectors
model1.wv.vectors.shape

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(X_train)]

model_ug_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, alpha=0.065, min_alpha=0.065)
model_ug_dbow.build_vocab(tagged_data)

similar_doc = model_ug_dbow.docvecs.most_similar('1')

len(model_ug_dbow.docvecs)

a = np.matrix(model_ug_dbow.docvecs[0])

for i in range(1,len(model_ug_dbow.docvecs)):
    a = np.vstack((a,model_ug_dbow.docvecs[i]))

a.shape

coln = ['X' + str(i) for i in range(100)]

X_train_new = pd.DataFrame(a, columns = coln)

SVM = svm.SVC(C=1.0, kernel='linear',degree = 3)
SVM.fit(X_train_new,Y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Y_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(predictions_SVM, Y_test)
