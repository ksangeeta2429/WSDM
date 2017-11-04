
# coding: utf-8

# In[37]:

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt


# In[34]:

train = pd.read_csv('../Data/train.csv')
songs = pd.read_csv('../Data/songs.csv')
test = pd.read_csv('../Data/test.csv')


# In[22]:

songs_in_train_and_test = np.intersect1d(train['song_id'].unique(), test['song_id'].unique())
shortlisted = np.union1d(train['song_id'].unique(), test['song_id'].unique())
#print shortlisted


# In[26]:

df = pd.DataFrame(songs)
new_songs = df.loc[df['song_id'].isin(shortlisted)]
#print len(songs) #2296320
#print len(new_songs) #384623


# In[29]:

new_songs.to_csv('shortlisted_song.csv')


# In[35]:

def split_str(string):
    multiple = re.split('/+|\|', string)
    return multiple

def brack_entry(string):
    a_1 = string[string.find("(")+1:string.find(")")]
    a_2 = string[0:string.find("(")-1]
    return a_1, a_2


# In[ ]:

headers = ['song_id', 'artist_name']
df_new = pd.DataFrame(columns=headers)
artists = new_songs['artist_name']
composers = new_songs['composer']
lyricists = new_songs['lyricist']

for row_index, row in new_songs.iterrows():
    artist = split_str(row['artist_name'])
    if len(artist) != 0:
        for i in range(len(artist)):
            df_new.loc[df_new.shape[0]] = [row['song_id'], artist[i]]
    else:
        df_new.loc[df_new.shape[0]] = [row['song_id'], row['artist_name']]


# In[41]:

for row_index, row in df_new.iterrows():
    string = row['artist_name']
    if (string.find("(") != -1):
        a1, a2 = brack_entry(row['artist_name'])
        df_new.append(pd.Series([row['song_id'], a1], headers), ignore_index=True)
        df_new.append(pd.Series([row['song_id'], a2], headers), ignore_index=True)
        df.drop([row_index])


# In[ ]:

new_songs.to_csv('artists.csv')

