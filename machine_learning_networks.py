# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
from xgboost import XGBClassifier


df = pd.read_csv('df.csv')
print(df.info())

df['senti'] = df['score'].map(lambda x: 'pos' if x > 0 else('neu' if x == 0 else 'neg'))

# down sampling
n_sample = 10000

df_1 = df[df['senti']=='pos']
df_01 = df[df['senti']=='neg']
df_0 = df[df['senti']=='neu']

downsampled = df_1.sample(n=n_sample, random_state=1)
df_01 = df_01.sample(n=n_sample, random_state=1)
df_10 = pd.concat([df_01, downsampled])
down = df_0.sample(n=n_sample, random_state=1)
df_down = pd.concat([down, df_10])
df_down.to_csv('downsampled.csv')

#change data type
df_down['clean'] = df_down['clean'].astype(str)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df_down["clean"].apply(lambda x: x.split(" ")))]
# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
# transform each document into a vector data
doc2vec_df = df_down["clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
df = pd.concat([df_down, doc2vec_df], axis=1)

# add tf-idfs columns
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(df["clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)

label = "senti"
ignore_cols = [label, "score", "Unnamed: 0", "Unnamed: 0.1",
               "Message-ID", "Date","From","To","Subject",
               "Text","score","entities","clean","key_phrase"]
features = [c for c in df.columns if c not in ignore_cols]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size = 0.20, random_state = 42)

# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)

print(feature_importances_df.head(2))

y_train_pred = rf.predict(X_train)

y_score = rf.predict(X_test)
y_test = y_test.tolist()
n_right = 0
for i in range(len(y_score)):
    if y_score[i] == y_test[i]:
        n_right += 1
        
print("Accuracy: %.2f%%" % ((n_right/float(len(y_test)) * 100)))
print(f1_score(y_test, y_score, average = "weighted"))
print(precision_score(y_test, y_score, average = "weighted"))
print(recall_score(y_test, y_score, average = "weighted")) 



xg = XGBClassifier(n_estimators = 100, random_state = 42)
xg.fit(X_train, y_train)

y_score = xg.predict(X_test)
n_right = 0
for i in range(len(y_score)):
    if y_score[i] == y_test[i]:
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(y_test)) * 100)))
print(f1_score(y_test, y_score, average = 'weighted'))
print(precision_score(y_test, y_score, average = 'weighted'))
print(recall_score(y_test, y_score, average = 'weighted')) 



dt = DecisionTreeClassifier(random_state = 42)
dt.fit(X_train, y_train)
y_score = dt.predict(X_test)
n_right = 0
for i in range(len(y_score)):
    if y_score[i] == y_test[i]:
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(y_test)) * 100)))
print(f1_score(y_test, y_score, average = 'weighted'))
print(precision_score(y_test, y_score, average = 'weighted'))
print(recall_score(y_test, y_score, average = 'weighted')) 


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
n_right = 0
for i in range(len(y_score)):
    if y_score[i] == y_test[i]:
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(y_test)) * 100)))
print(f1_score(y_test, y_score, average = 'weighted'))
print(precision_score(y_test, y_score, average = 'weighted'))
print(recall_score(y_test, y_score, average = 'weighted')) 








