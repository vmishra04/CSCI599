#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 19:49:45 2018

@author: vivekmishra
"""
#PAth
import os
os.chdir('/Users/vivekmishra/Desktop/USC/599-DSS/project')


#imports

import pandas as pd
import numpy as np
import re
import string
import unicodedata
import seaborn as sns
import matplotlib as plt
from nltk.stem import PorterStemmer
from nltk.corpus import words
from sklearn.cluster import KMeans,AgglomerativeClustering
import scipy


import nltk
nltk.download('words')
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')
blacklist = ["say","video","news","latest","youtube","watch","said","make","get","new","refinery",
             "go","get"]
#stopword_list.extend(blacklist)
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import spacy
nlp = spacy.load('en', parse=True, tag=True, entity=True)

#recommendation class
from recommendation import recommendation

#read pickle - Contains LDA And sentiment analysis results
df = pd.read_pickle('df_senti.pkl')



#Preproc func

def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_hashtag(text):
    entity_prefixes = ['#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    word_list = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                word_list.append(word)
    return ' '.join(word_list)

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

def get_verbs_and_nouns(sent):
   verbs = ""
   tokenized = sent_tokenize(sent)
   for i in tokenized:
       wordsList = nltk.word_tokenize(i)
       wordsList = [w for w in wordsList if not w in stopword_list]
       tagged = nltk.pos_tag(wordsList)
       for j in tagged:
           if j[1].startswith("VB"):
               verbs = verbs+" "+(j[0])
           elif j[1].startswith("NN"):
               verbs = verbs+" "+(j[0])
   if verbs!= "":
       return verbs
   else:
       return sent


#TF-IDF for title and desc - features into models
title = list(df['title'])

counter = 0
for sent in title:
    title[counter]=strip_links(sent)
    counter += 1
    
counter = 0
for sent in title:
    title[counter]=strip_hashtag(sent)
    counter += 1

counter = 0
for sent in title:
    title[counter]=remove_special_characters(sent, 
                          remove_digits=True)
    counter += 1
    
counter = 0
for sent in title:
    title[counter]=remove_stopwords(sent)
    counter += 1

counter = 0
for sent in title:
    title[counter]=lemmatize_text(sent)
    counter += 1
    
#counter = 0
#for sent in title:
#    title[counter]=get_verbs_and_nouns(sent)
#    counter += 1
    
vectorizer = TfidfVectorizer(strip_accents='unicode')
title_mat = vectorizer.fit_transform(title)
title_mat = title_mat.toarray()

title_mat = pd.DataFrame(title_mat)

#Desc
desc = list(df['desc'])

counter = 0
for sent in desc:
    desc[counter]=strip_links(sent)
    counter += 1
    
counter = 0
for sent in desc:
    desc[counter]=strip_hashtag(sent)
    counter += 1

counter = 0
for sent in desc:
    desc[counter]=remove_special_characters(sent, 
                          remove_digits=True)
    counter += 1
    
counter = 0
for sent in desc:
    desc[counter]=remove_stopwords(sent)
    counter += 1
    
counter = 0
for sent in desc:
    desc[counter]=lemmatize_text(sent)
    counter += 1
    
#Joining desc and title to form a word dictionary

word_dict = []

counter = 0
for text in title:
    tokens_title = tokenizer.tokenize(text)
    tokens_title = [token.strip() for token in tokens_title] 
    
    desc_text = desc[counter]
    tokens_desc = tokenizer.tokenize(desc_text)
    tokens_desc = [token.strip() for token in tokens_desc]
    
    merge = tokens_title+tokens_desc
    word_list = set()
    for item in merge:
        word_list.add(item)
        
    word_dict.append(list(word_list))
    
    counter += 1
    
counter = 0
for item in word_dict:
    word_dict[counter] = ' '.join(item)
    counter += 1    
    
#Subtitle topic 1
subt = list(df['topic1']) 

counter = 0
for item in subt:
    subt[counter] = ' '.join(item)
    counter += 1
       
#Tags
tags = list(df['tags'])
counter = 0
for item in tags:
    if isinstance(item,list):
       tags[counter] = ' '.join(item) 
    else:    
        item = "No tags"
        tags[counter] = item
    counter += 1 

counter = 0
for sent in tags:
    tags[counter]=strip_links(sent)
    counter += 1
    
counter = 0
for sent in tags:
    tags[counter]=strip_hashtag(sent)
    counter += 1

counter = 0
for sent in tags:
    tags[counter]=remove_special_characters(sent, 
                          remove_digits=True)
    counter += 1
    
counter = 0
for sent in tags:
    tags[counter]=remove_stopwords(sent)
    counter += 1
    
counter = 0
for sent in tags:
    tags[counter]=lemmatize_text(sent)
    counter += 1

    
    
####TF-IDF    
    
vectorizer = TfidfVectorizer(strip_accents='unicode')
word_mat = vectorizer.fit_transform(word_dict)
word_mat = word_mat.toarray()

word_mat = pd.DataFrame(word_mat)


#Feature Selection - Experimentation
## For time being only use title matrix
vectorizer = TfidfVectorizer(strip_accents='unicode')
title_mat = vectorizer.fit_transform(title)
title_mat = title_mat.toarray()
feature_mat = title_mat
feature_df = pd.DataFrame(title_mat)

#######Clustering

#Adding some more features to tf-idf matrix - scaling required

#title_mat['likes'] = df['likes']
#title_mat['dislike'] = df['dislike']
#title_mat['comment'] = df['comment']
#title_mat['senti_title'] = df['senti_title']
#title_mat['senti_desc'] = df['senti_desc']
#title_mat['senti_subt'] = df['senti_subt']

#Conversion of dataframe to spare matrix

no_of_cluster = 7

dense_matrix = np.array(feature_df.as_matrix(columns = None), dtype=bool).astype(np.int)
sparse_matrix = scipy.sparse.csr_matrix(dense_matrix)
kmeans = KMeans(n_clusters=no_of_cluster, random_state=0)
kmeans.fit(sparse_matrix)

#Evaluation of clustering

print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(no_of_cluster):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
    
clusters = kmeans.labels_.tolist()

#Counter for each cluster - To check cluster distribution
from collections import Counter
el = Counter(clusters)

#Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
dist = cosine_similarity(feature_mat)


#Silhouette  score - Best if closer to 1
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(feature_mat, kmeans.labels_)
print("Silhouette score " + str(silhouette_avg))

#Calinski-Harabaz Index¶ -  Higher the score better
from sklearn.metrics import calinski_harabaz_score
print("Calinski score : " + str(calinski_harabaz_score(feature_mat,clusters)))

#Davies-Bouldin Index - Closer to zero better
from sklearn.metrics import davies_bouldin_score
print("Davies-Bouldin score : " + str(davies_bouldin_score(feature_mat,clusters)))


#Recommendation
#Testing with video id
df = df.reset_index()
df = df.drop(['index'],axis=1)
vid = 'iUdgD8kYU-E' #Eg 1 Supreme court justice
vid = 'tG3wqbEmb7s' #eg 2 Iran nuclear deal
vid = 'Oms5r6_yJB8' #eg 3 Robert Mueller
feature_df['id'] = df['id']
feature_df['clusters'] = clusters
df['clusters'] = clusters
rec_obj = recommendation()
least_rel,most_rel = rec_obj.getRecommendation(vid,df,feature_df)

print("The titles of most relevent recommendation")

for item in most_rel:
    title_str = df[df['id']== item[0]]['title']
    print('Title: ' + str(title_str))
    
print("The title of least relevent recommendation")

for item in least_rel:
    title_str = df[df['id']== item[0]]['title']
    print('Title: ' + str(title_str))  
    
##################################################################################################
    
    
#Evaluation of clustering
from collections import defaultdict
cluster_terms = defaultdict(list)

print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(no_of_cluster):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        cluster_terms[i].append(terms[ind].lower())
        print(' %s' % terms[ind]),
    print
    
def plot_graph(random_state_seed):
    
    from sklearn.manifold import MDS
    
    MDS()
    
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state_seed)
    
    indexs_to_drop = []
    relavant_terms = []
    for k,v in cluster_terms.items():
        relavant_terms.extend(v)
    
    # cluster words:
    for i in range(len(title)):
        flag= True
        for j in title[i].split():
            if j.lower() in relavant_terms:
                flag = False
                break
        if flag:
            indexs_to_drop.append(i)
    
    dist = 1- cosine_similarity(feature_mat)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    
    xs, ys = pos[:, 0], pos[:, 1]
    
    for x in range(len(xs)):
        xs[x] = xs[x]*30.0
    
    for y in range(len(ys)):
        ys[y] = ys[y]*30.0
        
    print()
    print()
    
    
    #set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#ffff00', 6: '#ff00ff', 7: '#0f0fff',}
    
    #set up cluster names using a dict
    cluster_names = {0: 'Saudi Arabia Khashoggi Iran Journalist Prince', 
                     1: 'China Uncensor Chinese War Trump', 
                     2: 'refinery world make get youtube new pbs people live ucl', 
                     3: 'visualpolitik oil argentina chinese world trump',
                     4: 'president obama speaks first lady michelle white house address',
                     5: 'Hate post mail Washington Attack Japan Adpocalypse People Israel',
                     6: 'Kavanaugh Sanders Sarah Hillary'}
    
    
    print(cluster_terms.values)
    # remove non relavant clusters

    
    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title = title)) 
    print(len(df))
    for i, row in df.iterrows():
        if any(words in relavant_terms for words in row['title'].split()):
           continue
        else:
            df.drop(i, inplace = True)
    print(len(df))
    
    #group by cluster
    groups = df.groupby('label')
    
    # set up plot
    fig, ax = plt.subplots(figsize=(30, 15)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    #ax.margins(x=0, y=-0.25)
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    list_x = []
    for name, group in groups:
        ax.plot(group.x[:100], group.y[:100], marker='o', linestyle='', ms=12, 
                label=cluster_names[name], color=cluster_colors[name], 
                mec='none')
        list_x.extend(group.x[:100])
        ax.set_aspect('auto')
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
        
    ax.legend(numpoints=11)  #show legend with only 1 point
    
    #add label in x,y position with the label as the film title
    j = 0
    for i in range(len(df)):
            if df.iloc[i]['x'] in list_x:
                ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'][:50], size=10)  
    
        
        
    plt.show() #show the plot
    return
    #uncomment the below to save the plot if need be
    #plt.savefig('clusters_small_noaxes.png', dpi=200)
    ##########################################################################################


    
    
    
    
    
""" Alternative approach - Agglomerative clustering"""

    
no_of_cluster = 25

dense_matrix = np.array(feature_df.as_matrix(columns = None), dtype=bool).astype(np.int)
sparse_matrix = scipy.sparse.csr_matrix(dense_matrix)
kmeans = AgglomerativeClustering(n_clusters=no_of_cluster)
kmeans.fit(dense_matrix)

#Evaluation of clustering


    
clusters = kmeans.labels_.tolist()

#Counter for each cluster - To check cluster distribution
from collections import Counter
el = Counter(clusters)

#Cosine similarity

from sklearn.metrics.pairwise import cosine_similarity
dist = cosine_similarity(feature_mat)


#Silhouette  score - Best if closer to 1
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(feature_mat, kmeans.labels_)
print("Silhouette score " + str(silhouette_avg))

#Calinski-Harabaz Index¶ -  Higher the score better
from sklearn.metrics import calinski_harabaz_score
print("Calinski score : " + str(calinski_harabaz_score(feature_mat,clusters)))

#Davies-Bouldin Index - Closer to zero better
from sklearn.metrics import davies_bouldin_score
print("Davies-Bouldin score : " + str(davies_bouldin_score(feature_mat,clusters)))


#Recommendation
#Testing with video id
df = df.reset_index()
df = df.drop(['index'],axis=1)
vid = 'iUdgD8kYU-E' #Eg 1 Supreme court justice
vid = 'tG3wqbEmb7s' #eg 2 Iran nuclear deal
vid = 'Oms5r6_yJB8' #eg 3 Robert Mueller
feature_df['id'] = df['id']
feature_df['clusters'] = clusters
df['clusters'] = clusters
rec_obj = recommendation()
least_rel,most_rel = rec_obj.getRecommendation(vid,df,feature_df)

print("The titles of most relevent recommendation")

for item in most_rel:
    title_str = df[df['id']== item[0]]['title']
    print('Title: ' + str(title_str))
    
print("The title of least relevent recommendation")

for item in least_rel:
    title_str = df[df['id']== item[0]]['title']
    print('Title: ' + str(title_str))  


#temp code
def hasKev(sent):
    if ("kavanaugh" in sent) or ("Kavanaugh" in sent):
        return True
    else:
        return False
    
df['hasKav'] = df['title'].apply(lambda x : hasKev(x))

temp = df[df['hasKav'] == True]

