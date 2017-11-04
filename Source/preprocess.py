
# coding: utf-8

# In[ ]:

import numpy as np
import codecs
import math
import pandas as pd
from googletrans import Translator
import goslate
from langdetect import detect
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


# In[ ]:

songs = pd.read_csv('song_mod.csv', usecols = ['artist_name', 'composer', 'lyricist']).astype(str)
df = pd.DataFrame(songs)


# In[ ]:

def preprocess_str(string):
    if (string.find("(") == -1):
        return string
    else:
        a_1 = string[string.find("(")+1:string.find(")")]
        a_2 = string[0:string.find("(")-1]
        try:
            lang = detect(a_2)
            if(lang == 'en'):
                return a_2
            else:
                return a_1
        except:
            return string


# In[ ]:

headers = ['artist_name', 'composer', 'lyricist']
for i in range(len(songs)):
    row = df.iloc[i]
    artist = preprocess_str(row['artist_name'])
    composer = preprocess_str(row['composer'])
    lyricist = preprocess_str(row['lyricist'])
    try:
        if (detect(artist) == 'en' and detect(composer) == 'en' and detect(lyricist) == 'en'):
            row = pd.Series([artist, composer, lyricist], headers)
        else:
            df.drop(i)
    except:
        df.drop(i)


# In[ ]:

df.to_csv('song_mod1.csv', encoding='utf-8')

