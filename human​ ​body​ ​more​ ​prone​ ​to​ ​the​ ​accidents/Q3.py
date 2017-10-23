
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 22:24:21 2017

@author: andiapps
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:38:05 2017

@author: andiapps
"""
import pandas as pd
import os
import json
import string
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk import word_tokenize, pos_tag, ne_chunk
import re

file = 'osha.xlsx'

pf=pd.read_excel(file)#, sheetname='Sheet1')
print(pf)
pf.head()
pd.DataFrame(pf) #Converting to a Dataframe

types=pf['type']
prior=pf['prior']
bodypart=pf['bodypart']
unwanted=pf['unwanted']

#Tokenization
body_tokens=[]
for i in range(len(bodypart)):
    b_tokenize = word_tokenize(bodypart[i])
    body_tokens.append(b_tokenize)
    
    
types_tokens=[]
#del types_tokens
for i in range(3000):
     t_tokenize = word_tokenize(types[i])
     types_tokens.append(t_tokenize)
   
#Stop word removal and lemmatization
stop = stopwords.words('english')
snowball = nltk.SnowballStemmer('english')
#wnl = nltk.WordNetLemmatizer()

def preprocess(toks):
    toks = [ t.lower() for t in toks if t not in string.punctuation ]
    toks = [t for t in toks if t not in stop ]
    toks = [ snowball.stem(t) for t in toks ]
#    toks = [ wnl.lemmatize(t) for t in toks ]
    toks_clean = [ t for t in toks if len(t) >= 3 ]
    return toks_clean

body_clean = [ preprocess(f) for f in body_tokens]
body_flat = [ c for l in body_clean for c in l ]


fd_body = FreqDist(body_flat)

human_part=['Ligaments','Mouth','Teeth','Tongue','Liver','face','knee','abdomen','legs','stomach','small intestine','large intestine','lungs','kidney','brain','eye','ankle','chest','hip','pelvis','spine','neck','femur','face','head','arm','shoulder','pelvic','clavical','wrist','hand','finger','back','buttock','heart','palm']
human=[]
for l in body_flat: 
    if l in human_part:
        human.append(l)

count=FreqDist(human)

#Convert list to dataframe
h = pd.DataFrame({'col':human})
h.to_csv("Q3result.csv",sep=",")
    

import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt

textconvert=" ".join(human)
wc = WordCloud(background_color="white").generate(textconvert)
plt.axis("off")
plt.imshow(wc, interpolation='bilinear')
plt.show()
