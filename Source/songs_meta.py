
# coding: utf-8

# In[2]:

import gc
import numpy as np
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[3]:

songs = pd.read_csv('../Data/songs.csv')


# In[19]:

cols = list(songs.columns)
artists = pd.Series(songs.artist_name.sort_values(inplace=False).unique())
print len(artists) #222363
songs['alen'] = songs['artist_name'].apply(len)
print songs['alen'].max() #35010

composers = pd.Series(songs.composer.sort_values(inplace=False).unique())
print len(composers) #329825
songs['clen'] = songs['composer'].map(str).apply(len)
print songs['clen'].max() #7117

lyricists = pd.Series(songs.lyricist.sort_values(inplace=False).unique())
print len(lyricists) #110927
songs['llen'] = songs['lyricist'].map(str).apply(len)
print songs['llen'].max() #410


# In[22]:

for artist in songs['artist_name']:
    print artist


# In[ ]:



