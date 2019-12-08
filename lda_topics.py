# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas as pd 
from nltk.stem import WordNetLemmatizer
import string, re

from nltk.corpus import stopwords 

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

#load the data
df = pd.read_csv('downsampled.csv')

print(df.info())

# conduct data cleaning
def clean(text):
    stop = set(stopwords.words('english'))
    stop.update(("to","cc","subject","http","from","sent","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","origin","x","b"))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    # porter= PorterStemmer()
    
    text=text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and len(i) > 2 and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    #stem = " ".join(porter.stem(token) for token in normalized.split())
    
    return normalized

text_clean=[]
for text in df['Text']:
    text_clean.append(clean(text).split())
    
dictionary = corpora.Dictionary(text_clean)
text_term_matrix = [dictionary.doc2bow(text) for text in text_clean]

# build up lda model
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(text_term_matrix, num_topics=4, id2word = dictionary, passes=30)

# top tpoics in the dataset
print(ldamodel.print_topics(num_topics=4, num_words=10))


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_clean, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
pyLDAvis.enable_notebook(sort=True)
vis = pyLDAvis.gensim.prepare(ldamodel, text_term_matrix, dictionary)


#save to html file
pyLDAvis.save_html(vis, 'lda.html')

