# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
stem = PorterStemmer()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
stop_words = set(stopwords.words("english"))
from sklearn.feature_extraction.text import CountVectorizer


#load UCB labeled data
df = pd.read_csv('labeled.csv')
print(df.head(2))

def tokenizer(text):
    return word_tokenize(text)

df['clean'] = df['clean'].astype(str)

df['tokens'] = df['clean'].apply(lambda x: tokenizer(x))

vocab = [word for t in df['tokens'] for word in t]
vocab = [word for word in vocab if len(word)>1]
counter = Counter(vocab)
vocab = [word for word, count in counter.most_common(3000)]

text = df['tokens'].tolist()
documents = [' '.join(tokens) for tokens in df['tokens']]
corpus = documents



cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)
list(cv.vocabulary_.keys())[:10]
#Most frequently occuring words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
top_df.to_csv('top_words.csv')



#pip install wordcloud
#Word cloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(corpus))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)



