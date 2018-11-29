#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:27:29 2018

@author: vivekmishra
"""

import os
os.chdir('/Users/vivekmishra/Desktop/USC/599-DSS/StanceDataset')
import requests


import pandas as pd
import re
import string
import unicodedata
import seaborn as sns
import matplotlib as plt
#import emoji
from nltk.stem import PorterStemmer
from nltk.corpus import words

import preprocessor as p
from senti import senti


import nltk
nltk.download('words')
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')

import spacy
nlp = spacy.load('en', parse=True, tag=True, entity=True)

from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('train_ch.csv')
df_test = pd.read_csv('test.csv')

df = df.append(df_test)

tweet = list(df['Tweet'])

def remove_hashtag(input_text):
    return re.sub(r'(\s)#\w+', '', input_text) 
                  
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def remove_at(input_text):
    return re.sub(r'(\s)@\w+', '', input_text) 

def preproc(sent):
    return p.clean(sent)

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

    
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, ' ', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

    
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    whitelist = ["n't","not", "no"]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if (token not in stopword_list or token in whitelist)]
    else:
        filtered_tokens = [token for token in tokens if (token.lower() not in stopword_list or token in whitelist)]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#counter = 0
#for sent in tweet:
#    tweet[counter]=remove_hashtag(sent)
#    counter += 1
    
#counter = 0
#for sent in tweet:
#    print(sent)
#    tweet[counter]=remove_at(sent)
#    counter += 1

#counter = 0
#for sent in tweet:
#    tweet[counter]=strip_links(sent)
#    counter += 1
    
counter = 0
for sent in tweet:
    tweet[counter]=preproc(sent)
    counter += 1
    
counter = 0
for sent in tweet:
    tweet[counter]=remove_special_characters(sent, 
                          remove_digits=True)
    counter += 1
    
counter = 0
for sent in tweet:
    tweet[counter]=remove_stopwords(sent)
    counter += 1

counter = 0
for sent in tweet:
    tweet[counter]=lemmatize_text(sent)
    counter += 1
    
vectorizer = TfidfVectorizer(strip_accents='unicode')
tweet_mat = vectorizer.fit_transform(tweet)
tweet_mat = tweet_mat.toarray()

tweet_mat = pd.DataFrame(tweet_mat)


#Features
senti_obj = senti()
df['senti_tweet'] = df['Tweet'].apply(lambda x : senti_obj.main(x))

#Define target
target = list(df['Stance'])

counter = 0
for val in target:
    if val == 'AGAINST':
        target[counter] = 0
    elif val == 'FAVOR':
        target[counter] = 1
    else:
        target[counter] = 2
    
    counter += 1
    
tweet_mat['target'] = target
#Model
import xgboost as xgb 

y= tweet_mat['target'].values
X = tweet_mat.drop(['target'],axis=1).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#default parameters
params = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'multi:softprob',
}
params['eval_metric'] = "merror"
params['num_class'] = 3
num_boost_round = 999

#Hyperparameter tuning

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(6,8)
    for min_child_weight in range(4,6)
]
min_merror = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))

    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )

    # Update best MError
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (max_depth,min_child_weight)
        
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]
        
#tune subsample,colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(9,11)]
    for colsample in [i/10. for i in range(9,11)]
]  

min_merror = float("Inf")
best_params = None
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))

    # Update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )

    # Update best Merror
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = (subsample,colsample)
        
        
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]


min_merror = float("Inf")
best_params = None
for eta in [0.5,0.3, 0.03]:
    print("CV with eta={}".format(eta))

    # Update our parameters
    params['eta'] = eta

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=3,
        metrics={'merror'},
        early_stopping_rounds=10
    )

    # Update best Merror
    mean_merror = cv_results['test-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()
    print("\tMerror {} for {} rounds".format(mean_merror, boost_rounds))
    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params = eta
        
params['eta'] = best_params


model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

num_boost_round = model.best_iteration + 1
best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)   

best_model.save_model("my_model.model")

loaded_model = xgb.Booster()
loaded_model.load_model("my_model.model")

# And use it for predictions.
loaded_model.predict(dtest)


#Plots for poster
#df['Stance'].hist(by=df['Target'])

df = df.replace("Climate Change is a Real Concern", value="Climate Change")
df =df.replace("Legalization of Abortion", value="Abortion")

import seaborn as sns
import itertools
#sns.set(style="darkgrid")
ax = sns.countplot(y="Stance", hue="Target", data=df,palette="Paired",orient="v")
plt.lengend(loc="bottom")
fig = ax.get_figure()
fig.savefig("output.png")

palette = itertools.cycle(sns.color_palette("Paired"))
import matplotlib.pyplot as plt
#for i in range(1, 7):
fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1)
c= next(palette)
sns.distplot(df[df['Target'] == 'Hillary Clinton']['senti_tweet'],label='Clinton', color=c)
ax1.legend()

ax1 = fig.add_subplot(2, 3, 2)
c= next(palette)
sns.distplot(df[df['Target'] == 'Legalization of Abortion']['senti_tweet'],label='Abortion', color=c)
ax1.legend()

ax1 = fig.add_subplot(2, 3, 3)
c= next(palette)
sns.distplot(df[df['Target'] == 'Atheism']['senti_tweet'],label='Atheism', color=c)
ax1.legend()

ax1 = fig.add_subplot(2, 3, 4)
c= next(palette)
sns.distplot(df[df['Target'] == 'Climate Change is a Real Concern']['senti_tweet'],label='Climate', color=c)
ax1.legend()

ax1 = fig.add_subplot(2, 3, 5)
c= next(palette)
sns.distplot(df[df['Target'] == 'Feminist Movement']['senti_tweet'],label='Feminism', color=c)
ax1.legend()


ax1 = fig.add_subplot(2, 3, 6)
c= next(palette)
sns.distplot(df[df['Target'] == 'Donald Trump']['senti_tweet'],label='Trump', color=c)
ax1.legend()

fig.savefig('dist.png')





