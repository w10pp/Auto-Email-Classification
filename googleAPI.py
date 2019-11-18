#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon ----

@author: w10
"""
import pandas as pd

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/---.json"

language_client = language.LanguageServiceClient()


# calculate the sentiment scores
def get_sentiment_scores(content):
    try:
        document = types.Document(content=content, type=enums.Document.Type.PLAIN_TEXT)
        sentiments = language_client.analyze_sentiment(document=document)
        score = sentiments.document_sentiment.score
        magnitude = sentiments.document_sentiment.magnitude
        
    except:
        score = 'na'
        magnitude = 'nan'
        
    print(score, magnitude)
    
    return pd.Series([score, magnitude])

#load the data
df_1 = pd.read_csv('clean.csv')

print(df_1.head(2))

df_sentiscore = df_1.content.apply(get_sentiment_scores)

df_sentiscore = pd.DataFrame(df_sentiscore)
df_sentiscore.columns = ['score','magnitude']

df_senti = pd.concat([df_1, df_sentiscore], axis = 1)


def get_top_entities(content):
    document = types.Document(content=content, type=enums.Document.Type.PLAIN_TEXT)
    entities = language_client.analyze_entities(document=document)
    
    return ', '.join([e.name for e in entities.entities[:5]])

top_entities = df_1.content.apply(get_top_entities)


df_senti['entities'] = top_entities


print(df_senti.head(2),'finished!!')
df_senti.to_csv('df_sen_en.csv')
