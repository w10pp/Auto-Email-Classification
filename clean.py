#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from pathlib import Path

#====Text data cleanning====
# Collect the email content
def get_email_content(message):
    contents = []
    for content in message.walk():
        if content.get_content_type() == "text/plain":
            contents.append(content.get_payload().lower())
    
    return ''.join(contents) 

# Tokenization
def tokens(contents):
    return word_tokenize(contents)

# Stemming and lemmatizing
def stem_lemma(tokens):
    stem = SnowballStemmer('english')
    lemma = WordNetLemmatizer()
    return [stem.stem(lemma.lemmatize(token)) for token in tokens]

# Remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

# Remove stopwords ii
def get_stop_words(stop_file_path):
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stoplist = f.readlines()
        stop_set = list(m.strip() for m in stoplist)
        return stop_set

stoplist = get_stop_words(Path(r"/Users/yilinwang/Downloads/capstone/stopwords.txt"))

def remove_stoplist(text):
    text = ' '.join([word for word in text.split()
                            if word not in stoplist])
    return text


# Remove punctuation
def punc_remover(list_):
    filted = []
    for i in list_:
        filted.append(i)
        if i in string.punctuation or i == '--' or i == '``' or i == '.' or i == ':' or i == '//':
            filted.remove(i)
    
    return filted
     
# Combination
def clean_content(content):
    words = remove_stoplist(content)
    token = tokens(words)
    stem_lemmaed = stem_lemma(token)
    filted_data = punc_remover(stem_lemmaed)
    
    return filted_data




    


