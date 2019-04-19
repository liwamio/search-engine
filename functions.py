
# coding: utf-8

# In[ ]:


import pandas as pd 
import json
import numpy as np
from collections import defaultdict

#calculate Mean Reciprocal Ranking for queries 

def Mrr_ranking(q, a, s): 
    
    gr_truth_file_dir = 'DMT4BaS/hw_1/part_1/Cranfield_DATASET/cran_Ground_truth.tsv'
    search_file_dir = 'DMT4BaS/hw_1/part_1/'
    
    with open(search_file_dir + a + '_'+ s +'.json') as fp: 
            search_file = json.load(fp)

    truth_file = pd.read_csv(gr_truth_file_dir, sep = '\t')
    mrr = []
    for i in q:    
        
        relevant_docs = truth_file[truth_file['id'] == i]['doc_id'] #file containing the ground truth document id for query_i
        doc_returned = [v for k,v in search_file.items() if k == str(i)]  #Document ids returned using search engine x for query_i      
        '''
        For each returned document, check if the document exists inside ground truth file. 
        If it exists get the reciprocal of the index of the document from our search result. 
        After doing these for all queries, sum the reciprocal of the indexes of the documents found 
        in the ground truth file and divide by the length of the query. 
        '''        
        try: 
            for d_t in doc_returned[0]:                 
                if d_t in relevant_docs.tolist():  
                    mrr.append(1/(doc_returned[0].index(d_t)+1))  
                    break
        except: 
            continue    
    
    score =  sum(mrr)/len(mrr)   
    return score
    

'''
For each search configuration divide the length of the intersection 
of the search result and ground truth by length of the ground truth. 
'''
def r_precision(q, k): 
    
    file_truth_dir = 'DMT4BaS/hw_1/part_1/Cranfield_DATASET/cran_Ground_truth.tsv'
    search_file_dir = 'DMT4BaS/hw_1/part_1/'
    truth_file = pd.read_csv(file_truth_dir, sep='\t')
    r_prec = []    
    
    with open(search_file_dir + k +'.json') as fp: 
        search_file = json.load(fp)
    for q_id in q: 
        relevant_docs = truth_file[truth_file['id'] == q_id]['doc_id']
        doc_returned = [v for k,v in search_file.items() if k == str(q_id)]
        try: 
            r_prec.append(len(set(relevant_docs.tolist()).intersection(set(doc_returned[0])))/len(doc_returned[0]))
        except: 
            r_prec.append(0)        
            
    return r_prec    


def dcg_idcg(q,a,k): 
    search_file_dir = 'DMT4BaS/hw_1/part_1/'
    with open(search_file_dir + a +'.json') as fp: 
            search_file = json.load(fp)
    file_truth_dir = 'DMT4BaS/hw_1/part_1/Cranfield_DATASET/cran_Ground_truth.tsv'
    truth_file = pd.read_csv(file_truth_dir, sep='\t')
    
    d_i = []
    for q_id in q: 
        
        doc_returned = [v for k,v in search_file.items() if k == str(q_id)]
        relevant_docs = truth_file[truth_file['id'] == q_id]['doc_id']
        
        dcg = []
        idcg = []            
            
        for i in range(1,k+1):                        
            if doc_returned and doc_returned[0][i] in relevant_docs :                                     
                dcg.append(1/np.log2(i+1))                      
                idcg.append((2^1 - 1)/ np.log2(i+1))                    
            else:                 
                dcg.append(0/np.log2(i+1))
                idcg.append(0/np.log2(i+1))          
        if(sum(idcg) == 0):
            d_i.append(0)
        else:                
            d_i.append(sum(dcg)/sum(idcg))                   
        
    return d_i


def nDCG(q,k, search_configuration): 
    dicts = defaultdict()
    for a in range(len(search_configuration)):         
        for kk in range(1,k+1):                  
            dcg_at_kk = np.mean(dcg_idcg(q,search_configuration[a],kk))            
            #dicts[len(dicts)] = [search_configuration[a], kk,dcg_at_kk]                        
            try: 
                dicts[search_configuration[a]].append([kk, dcg_at_kk])
            except: 
                dicts[search_configuration[a]] =  [[kk, dcg_at_kk]]
                
    return dicts
    
    

