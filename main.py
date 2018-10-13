#!/usr/bin/env python3
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


directory = '/Users/vivekmishra/Desktop/USC/599-DSS/project/video/'
vid_ids = []
labels = []
mean_rgb = []
mean_audio = []

for filename in os.listdir(directory):
    if filename.endswith(".tfrecord"):

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

new_id = []
for vid_id in vid_ids:
    print(vid_id) 
    new_id.append(getVideoID(vid_id))
    




#Description of label
text = pd.read_csv('https://research.google.com/youtube8m/csv/2/vocabulary.csv')





