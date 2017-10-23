# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing necessary libraries
import nltk
from nltk import word_tokenize
from nltk import pos_tag
from nltk.chunk import *
from nltk.tokenize import sent_tokenize
import pandas as pd


source_file = 'Copy of osha.xlsx'

pf=pd.read_excel(source_file)  #sheetname='Sheet1'
pf.head()
pd.DataFrame(pf)        #Converting to a Dataframe

type_1=pf['type']
job_descriptions=pf['prior']
body=pf['bodypart']
unwanted=pf['unwanted']
final_list = []
missing_was = []
captured_was = []
for i in job_descriptions:
    i = i.lower()
    j = sent_tokenize(i)[0]
#    k = word_tokenize(j) 
    if not j.find(" was ") == -1:
        captured_was.append(j[j.index(" was ")+5:])
    elif not j.find(" were ") == -1:
        captured_was.append(j[j.index(" were ")+6:])
    else:
        missing_was.append(j)
activity_list = [] 
missing_activity = []
end_list = []
for i in captured_was:
    start = ''
    end = ''
    end_len = 0
    temp = word_tokenize(i)
    pos = pos_tag(temp)
    for j in pos:
        if j[1] == 'VBG':
            if not start:
                start = j[0]
        if start:
            if j[1] == 'NN' or j[1] == 'NNS':
                if not end:
                    end = j[0]
                    end_len = len(j[0])
                    end_list.append(end)
                    break
                elif j[1] == '.':
                    if not end:
                        end = j[0]
                        end_len = len(j[0])
                        end_list.append(end)
    temp1 = i[i.index(start):i.index(end)+end_len]
    if " " in temp1:
        activity_list.append(temp1)
    else:
        missing_activity.append(pos)
    
import pandas as pd
df = pd.DataFrame(activity_list,columns = ["Activities"])
df2= pd.DataFrame(end_list)
with open("activity.csv", "w") as o:
    df.to_csv(o)
with open("activity_nouns.csv", "w") as o:
    df2.to_csv(o)
import pandas
from collections import Counter
letter_counts = Counter(end_list)
df = pandas.DataFrame([v for x,v in letter_counts.most_common(50)], [x for x,v in letter_counts.most_common(50)]  )
df.plot(kind='bar')