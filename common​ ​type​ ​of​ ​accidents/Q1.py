# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:47:07 2017

@author: saipraveen
"""

import pandas as pd

#Set file path 
#MsiaAccidents.csv train data -- filename:MsiaAccidentCases.csv
df = pd.read_csv("MsiaAccidentCases.csv")
osha = pd.read_csv("osha.csv")
df.head()
length = df['SummaryCase'].apply(len)
df = df.assign(Length=length)
df.head()

x = df['SummaryCase'].values
y = df['Cause'].values
set(y)

#MsiaAccidents test data -- file name:test.csv
test = pd.read_csv("test.csv")


#Plot the distribution of the document length for each category
import matplotlib.pyplot as plt
df.hist(column='Length',by='Cause',bins=50)
plt.show()

#Frequency of words within each doc

import nltk
from nltk.corpus import stopwords
import string
#import re
#pattern=re.compile('r^(?:(?:[0-9]{2}[:\/,]){2}[0-9]{2,4}|am|pm)$'+r'|'.join(stopwords.words('english')))
newstopwords=stopwords.words("English") + ['yuhao','the','is','it','may','approximately','empployee'] 
WNlemma = nltk.WordNetLemmatizer()


def pre_process(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens=[WNlemma.lemmatize(t) for t in tokens]
    tokens=[word for word in tokens if word not in newstopwords]
    text_after_process=" ".join(tokens)
    return(text_after_process)

df['SummaryCase'] = df['SummaryCase'].apply(pre_process)
df.head()
length = df['SummaryCase'].apply(len)
df = df.assign(Length = length)
df.head()

#Preprocess test data
test_summ = test['summary'].apply(pre_process)
test_labels=test['cause']

#preprocess osha data
XX = osha['summary'].apply(pre_process)

#split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, label_train,label_test = train_test_split(df.SummaryCase, df.Cause , test_size=0.30, random_state=12)

#Create dtm by using word occurence
from sklearn.feature_extraction.text import CountVectorizer


count_vect = CountVectorizer( )

X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


count_vect.get_feature_names()


dtm1 = pd.DataFrame(X_train_counts.toarray().transpose(), index = count_vect.get_feature_names())
dtm1=dtm1.transpose()
dtm1.head()
dtm1.to_csv('dtm1.csv',sep=',')    


#Create dtm by using Term Frequency. 
#Divide the number of occurrences of each word in a document 
#by the total number of words in the document: 
#these new features are called tf for Term Frequencies
#If set use_idf=True, which mean create dtm by using tf_idf

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

#Building Modeling by using Naïve Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tf, label_train)

#Build a pipeline: Combine multiple steps into one
from sklearn.pipeline import Pipeline
text_clf_NB = Pipeline([('vect', CountVectorizer()),  
                     ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                    ])

#Use pipeline to train the model
text_clf_NB.fit(X_train,label_train )

#Test model accuracy
import numpy as np
from sklearn import metrics 
predicted = text_clf_NB.predict(X_test)
#np.mean(predicted == y_test) 
print(metrics.confusion_matrix(label_test, predicted))
print(np.mean(predicted == label_test) )
#Accuracy -- 51%


#Decision Tree
from sklearn import tree
text_clf_tree = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                      ('clf', tree.DecisionTreeClassifier())
                    ])
text_clf_tree.fit(X_train,label_train)
predicted = text_clf_tree.predict(X_test)

print(metrics.confusion_matrix(label_test, predicted))
print(np.mean(predicted == label_test) )
#Accuracy -- 60%


#SVM
from sklearn.linear_model import SGDClassifier
text_clf_SVM = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf', SGDClassifier(
                                            alpha=1e-3 
                                             ))
                    ])

text_clf_SVM.fit(X_train, label_train)
predicted = text_clf_SVM.predict(X_test)
print(metrics.confusion_matrix(label_test, predicted))
print(np.mean(predicted == label_test))   
print(metrics.classification_report(label_test, predicted))
#Accuracy -- 72%

####Test data 
predicted = text_clf_SVM.predict(test_summ)
print(metrics.confusion_matrix(test_labels, predicted))
print(np.mean(predicted == test_labels))   
print(metrics.classification_report(test_labels, predicted))
#Accuracy -- 71%

#SVM with stochastic gradient decent -- MODEL WITH HIGH Accuracy    
text_clf_SGD = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                      ('clf', SGDClassifier(alpha=1e-3, loss='log'))
                    ])
    
text_clf_SGD.fit(X_train, label_train) 
predicted = text_clf_SGD.predict(X_test)
print(metrics.confusion_matrix(label_test, predicted))
print(np.mean(predicted == label_test) )
print(metrics.classification_report(label_test, predicted))
#Accuracy -- 77%
####Test data
predicted = text_clf_SGD.predict(test_summ)
print(metrics.confusion_matrix(test_labels, predicted))
print(np.mean(predicted == test_labels))   
print(metrics.classification_report(test_labels, predicted))
#Accuracy -- 71%
###larger data -- osha dataset
#Use model to predict Osha data
osha_predict_sgd = text_clf_SGD.predict(XX)
#Merging predicted classes with osha data
osha = osha.assign(Cause=osha_predict_sgd)
osha.to_csv("output.csv",sep=",")

