from whoosh import index
from whoosh.qparser import *
from whoosh.query import *
from whoosh import scoring
import pandas as pd
import json
from collections import defaultdict
from nltk.corpus import stopwords
import nltk


dir_of_index = './directory_index_StemmingAnalyzer'
ix = index.open_dir(dir_of_index)

query_dir = './Cranfield_DATASET/cran_queries.tsv'
querys = pd.read_csv(query_dir, sep='\t')

scoring_function = scoring.BM25F(title_B = 0.8, body_B = 0.9, k1 = 1.5)
#

returned_docs = defaultdict()
stopwords = set(stopwords.words('english'))

for index, row in querys.iterrows():

    input_query = row['Query']
    qp = MultifieldParser(['body', 'title'], ix.schema, termclass = Variations)
    parsed_query = qp.parse(input_query)

    searcher = ix.searcher(weighting = scoring_function)
    results = searcher.search(parsed_query, limit = None)

    temp = [int(hit['id']) for hit in results]

    returned_docs[row['Query_ID']] = temp

with open('stemming_analyzer.json', 'w') as fp:
    json.dump(returned_docs, fp)

searcher.close()
