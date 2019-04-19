
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import functions as f
import json
import pandas as pd
import numpy as np
import json



#Iterate through each html file and extract `Title` and `Body` using BeautifulSoup and save it a csv file.

doc_dir = 'DMT4BaS/HW_1/part_1/Cranfield_DATASET/DOCUMENTS/______'
docs = pd.DataFrame(columns = ['Id', 'Title', 'Body'])

for i in range(1,1401):

    temp= []
    soup = BeautifulSoup(open(doc_dir + str(i) +'.html'), 'lxml')

    temp.append(i)
    temp.append(soup.find_all('title')[0].get_text())
    temp.append(soup.find_all('body')[0].get_text())

    docs.loc[len(docs)] = temp

docs.to_csv('DMT4BaS/HW_1/part_1/documents.csv', index = False, encoding = 'utf-8')

from collections import defaultdict

analyzers = ['stemming', 'simple', 'standrad', 'fancy']
scoring = ['bm25f', 'mw', 'tfidf', 'frequency']

summary = dict()

q = range(1,226)
#Loop through each search engine configuratoin (Analzyer + Scoring) to get the MRR(mean reciprocal ranking) value
index = 0
for i in range(len(analyzers)):
    for j in range(len(scoring)):
        index += 1

        score = f.Mrr_ranking(q, analyzers[i] , scoring[j])
        summary[index] = [analyzers[i] + '_' + scoring[j], score]


analy = ['Stemming', 'Simple', 'Standard', 'Fancy']
scor = ['BM25F(title_B = 0.8, body_B = 0.9, k1 = 1.5)', 'MultiWeighting(BM25F(), body = TF_IDF)', 'TF_IDF', 'Frequency']

schema_of_conf = defaultdict()
for i in analy:
    for j in scor:
        try:
            schema_of_conf[i].append(j)
        except:
            schema_of_conf[i] = [j]


analy = ['Stemming', 'Simple', 'Standard', 'Fancy']
scor = ['BM25F', 'MultiWeighting', 'TF_IDF', 'Frequency']
temp = []
for i in range(len(analyzers)):
    for j in range(len(scoring)):
        temp.append(analy[i] + ' Analyzer with ' + scor[j] + ' Scoring function.')

#convert our result from dictionary form to DataFrame
df = pd.DataFrame.from_dict(summary, orient='index', columns = ['Searching Configuration', 'Mrr'])
#replace column searching configuration with a descriptive value
df['Searching Configuration'] = temp
#sort
df = df.sort_values(['Mrr'], ascending = False).reset_index(drop = True)


q = range(1,226)

#select searching configuration with an mrr value greater or equal to 0.32
choosen_conf = dict()
for i in range(1,len(summary)+1):
    if summary[i][1] >= 0.32:
        choosen_conf[summary[i][0]] = f.r_precision(q, summary[i][0])


r_prec = pd.DataFrame(columns = ['Search Configuration', 'Mean', 'Min', '1st quartile', 'Median', '3rd quartile', 'Max'])

for i in choosen_conf:
    temp = []
    temp.append(i)
    temp.append(np.mean(choosen_conf[i]))
    temp.append(np.min(choosen_conf[i]))
    temp.append(np.percentile(choosen_conf[i], 25))
    temp.append(np.percentile(choosen_conf[i], 50))
    temp.append(np.percentile(choosen_conf[i], 75))
    temp.append(np.max(choosen_conf[i]))

    r_prec.loc[len(r_prec)] = temp

search_configuration = r_prec['Search Configuration'].tolist()
dcg_values = f.nDCG(range(1,226),10, search_configuration)



