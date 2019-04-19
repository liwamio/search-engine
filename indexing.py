from whoosh.index import create_in
from whoosh import index
from whoosh.fields import *
from whoosh.qparser import *
from whoosh.analysis import *
from whoosh import scoring
import csv, os
from nltk.corpus import stopwords


stopwords = set(stopwords.words('english'))
selected_analyzer = StemmingAnalyzer(expression ='\\w+', stoplist = stopwords, minsize = 3)| LowercaseFilter()

csv_doc_dir = 'documents.csv'

schema = Schema ( id = ID(stored = True), title = TEXT(stored = False, analyzer = selected_analyzer), body = TEXT(stored = False, analyzer = selected_analyzer, phrase = True))

dir_of_index = './directory_index_StemmingAnalyzer'

try:
    ix = create_in(dir_of_index, schema)
except:
    os.mkdir(dir_of_index)
    ix = create_in(dir_of_index, schema)

ix = index.open_dir(dir_of_index)

writer = ix.writer(procs = 1, limitmb = 500)

in_file = open(csv_doc_dir, 'r', encoding = 'utf-8')
csv_reader = csv.reader(in_file, delimiter = ',')
csv_reader.__next__()

for record in csv_reader:
    id = record[0]
    title = record[1]
    body = record[2]

    writer.add_document(id = id, title = title, body = body)

writer.commit()
in_file.close()

