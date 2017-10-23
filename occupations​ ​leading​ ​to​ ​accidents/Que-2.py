
#importing necessary libraries
import nltk
from nltk import word_tokenize
from nltk import pos_tag
from nltk.chunk import *
from nltk.tokenize import sent_tokenize
import pandas as pd


#rules for chunking
job_grammar = r"""
  job: {<NN>+<CC><NN><VBD><VBG>} 
       {<DT><NN><VBD><VBG>}
       {<DT><NN>+<CC>}
       {<CD><DT><NN><RB>}
       {<CD><DT><JJ>?<NN>+<VBD>}
       {<CD><DT><JJ>?<NN>+<IN>}
       {<NNP><NN><VBD>}
       {<DT><JJ>?<NN><VBN>}
       {<VBD><DT><JJ>?<NN><VBG>}
       {<JJ><NN>+<IN>}


"""
#       {<VBG><IN><DT><NN><IN>}



source_file = 'Data/Copy of osha.xlsx'

pf=pd.read_excel(source_file)  #sheetname='Sheet1'
pf.head()
pd.DataFrame(pf)        #Converting to a Dataframe

type_1=pf['type']
job_descriptions=pf['prior']
body=pf['bodypart']
unwanted=pf['unwanted']
final_list = []

for sentences in job_descriptions:
    s_list = sent_tokenize(sentences)
    sentences = s_list[0]
    pos = pos_tag(word_tokenize(sentences))
    chunker_obj = nltk.RegexpParser(job_grammar)
    chunker_result = chunker_obj.parse(pos)
#    chunker_result.draw()
    word_list= []
    for word in chunker_result:
        if type(word) is nltk.Tree and word.label() == 'job':
            for i in word.leaves():
                if i[1] == 'NN':
                    word_list.append(i[0])
    f_word = " ".join(x for x in word_list)
    list1 = f_word.split(' ')
    if 'operator' in list1:
        index = list1.index('operator')
    elif  'laborer' in list1:
        index = list1.index('laborer')
    elif  'driver' in list1:
        index = list1.index('driver')
    elif  'mechanic' in list1:
        index = list1.index('mechanic')
    elif  'technician' in list1:
        index = list1.index('technician')
    elif  'technician' in list1:
        index = list1.index('technician')
    else:
        index = 0
    
    if(index>0):
        f_word1 = list1[index-1] + " " + list1[index]    
    else:
        f_word1 = f_word
    
    if f_word1:
        print(f_word1)
        final_list.append(f_word1) 
        
        


import pandas
from collections import Counter
letter_counts = Counter(final_list)
letter_counts.pop("male")
letter_counts.pop("truck")
df = pandas.DataFrame([v for x,v in letter_counts.most_common(50)], [x for x,v in letter_counts.most_common(50)]  )
df.plot(kind='bar')
df = pandas.DataFrame([v for x,v in letter_counts.most_common(100)], [x for x,v in letter_counts.most_common(100)]  )
with open("occupation.csv", "w") as o:
    df.to_csv(o)

for letter, count in letter_counts.most_common(50):
    print('%s: %7d' % (letter, count))