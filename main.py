CapDownload#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 20:10:44 2018

@author: vivekmishra
"""

import os
os.chdir('/Users/vivekmishra/Desktop/USC/599-DSS/project')
import requests


import tensorflow as tf
import numpy as np
import pandas as pd
from IPython.display import YouTubeVideo

from yt_api import Description
#from cap import CapDownload
from alternateCap import alternateCap


directory = '/Users/vivekmishra/Desktop/USC/599-DSS/project/video/'
vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for filename in os.listdir(directory):
    if filename.endswith(".tfrecord"):
        if filename.startswith("train"):
            video_lvl_record = directory+'/'+filename
            for example in tf.python_io.tf_record_iterator(video_lvl_record):
                tf_example = tf.train.Example.FromString(example)
        
                vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
                labels.append(tf_example.features.feature['labels'].int64_list.value)
                mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
                mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
    
    

def getVideoID(id):
    url = 'http://data.yt8m.org/2/j/i/'+id[0]+id[1]+'/'+id+'.js'
    r = requests.get(url)
    data = r.text
    arr = data.split(",")
    if len(arr) > 1:
        new_id = arr[1]
        new_id = new_id.strip('"')
        new_id = new_id.rstrip('");')
        return new_id
    else:
        return np.nan
    
####Analysis - Not required
print('Number of videos in this tfrecord: ',len(mean_rgb))
print('First video feature length',len(mean_rgb[0]))
print('First 20 features of the first youtube video (',vid_ids[0],')')
print(mean_rgb[0][:20])

def play_one_vid(record_name, video_index):
    id = vid_ids[video_index]
    new_id = getVideoID(id)
    return new_id
    
# Show Video
YouTubeVideo(play_one_vid(video_lvl_record, 0))

####

### From this part we create the main dataframe for sentiment analysis part
### DF contains video_id,description,title,likes,views for now

df = pd.DataFrame()



#Get Video_id for each video
new_id = []
count = 1
for vid_id in vid_ids:
    print(count) 
    new_id.append(getVideoID(vid_id))
    count = count+1
       
df['id'] = new_id
df.to_pickle("df.pkl")

#
df = pd.read_pickle("df_new.pkl")
new_id = df['id']
    
#Get Desc,Title, Tags, Views
#Description of label
text = pd.read_csv('https://research.google.com/youtube8m/csv/2/vocabulary.csv')
description = Description()

desc_data = []
title = []
caption = []
views = []
likes = []
dislike = []
favorite = []
comment = []
count = 1
desc_id = []
tags = []
label_tag = []
index = 0
for vid_id in new_id:
    #client = description.get_authenticated_service()
    print(count)
    data = description.videos_list_by_id(client,part='snippet,contentDetails,statistics',id=vid_id)
    if 'items' in data.keys():
        if len(data['items']) > 0:
            desc_id.append(vid_id)
            label_tag.append([text['Name'][id] for id in labels[index]])
            desc_data.append(data['items'][0]['snippet']['description'])  
            title.append(data['items'][0]['snippet']['title'])
            caption.append(data['items'][0]['contentDetails']['caption'])
            
            if 'tags' in data['items'][0]['snippet'].keys():
                tags.append(data['items'][0]['snippet']['tags'])
            else:
                tags.append(np.nan)
            
            if 'viewCount' in data['items'][0]['statistics'].keys():
                views.append(int(data['items'][0]['statistics']['viewCount']))
            else:
                views.append(np.nan)
                
            if 'likeCount' in data['items'][0]['statistics'].keys():
                likes.append(int(data['items'][0]['statistics']['likeCount']))
            else:
                likes.append(np.nan)
                
            if 'dislikeCount' in data['items'][0]['statistics'].keys():   
                dislike.append(int(data['items'][0]['statistics']['dislikeCount']))
            else:
                dislike.append(np.nan)
                
            if 'favoriteCount'  in  data['items'][0]['statistics'].keys():  
                favorite.append(int(data['items'][0]['statistics']['favoriteCount']))
            else:
                favorite.append(np.nan)
                
            if 'commentCount' in  data['items'][0]['statistics'].keys():  
                comment.append(int(data['items'][0]['statistics']['commentCount']))
            else:
                comment.append(np.nan)
                
            count = count+1
    index+=1
    


df['id'] = desc_id
df['title'] = title
df['desc'] = desc_data
df['views'] = views
df['likes'] = likes
df['dislike'] = dislike
df['favorite'] = favorite
df['comment'] = comment
df['tags'] = tags
df['caption'] = caption
df['labels'] = label_tag


df.to_pickle("df_new.pkl")
df.to_csv("df_new.csv")



#Get List of Captions + download

cap = alternateCap() 
head['subtitle'] = head.apply(lambda row: cap.downloadCap(row['id']) if row['caption'] == 'true' 
                                else np.nan,axis=1)













