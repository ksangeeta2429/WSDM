
# coding: utf-8

# In[1]:

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


# In[2]:

train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')
songs = pd.read_csv('../Data/songs.csv')
members = pd.read_csv('../Data/members.csv')
#songs_meta = pd.read_csv('../Data/song_extra_info.csv')


# In[ ]:

#print train.iloc[0] 
#print songs.iloc[0]
#print members.iloc[0]


# In[3]:

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)

members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')

train = train.fillna(-1)
test = test.fillna(-1)

#print train.iloc[0]


# In[4]:

cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        #msno 30755
        #song_id 359966
        #source_system_tab 10
        #source_screen_name 21
        #source_type 13
        #artist_name = 40583
        #genre_ids = 573
        #gender = 3
        #song_length language city bdregistered_via expiration_date registration_year registration_month registration_date expiration_year expiration_month
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])


# In[5]:

X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values
print X_test


# In[7]:

cols_name = list(train.columns)
X_new = X

max_vals = X_new.max(axis = 0).transpose()
min_vals = X_new.min(axis = 0).transpose()
mean_vals = np.mean(X_new, axis = 0).transpose()

#training set
X_new = X_new - mean_vals
X_new = X_new / (max_vals - min_vals)
X_new = np.around(X_new,decimals = 2)

#testing set
X_new_test = X_test - mean_vals
X_new_test = X_new_test / (max_vals - min_vals)
X_new_test = np.around(X_new_test,decimals = 2)
print X_new_test.max(axis = 0)
print X_new_test.min(axis = 0)

print X_new.max(axis = 0)
print X_new.min(axis = 0)


# In[8]:

np.savetxt("train.csv", X_new, delimiter=",")
np.savetxt("test.csv", X_new_test, delimiter=",")


# In[9]:

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state = 10)
    
del X, y; 
del members, songs;
del train, test;
gc.collect();


# In[ ]:

model = Sequential([
    Dense(units=1024, kernel_initializer='uniform', input_dim=19, activation='relu'),
    Dense(units=512, kernel_initializer='uniform', activation='relu'),
    Dropout(0.25),
    Dense(128, kernel_initializer='uniform', activation='relu'),
    Dense(64, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=20)


# In[ ]:

score = model.evaluate(X_validation, y_validation, batch_size=X_validation.shape[0])
print '\nLoss is ', score[0]
print '\nAnd the Score is ', score[1] * 100, '%'


# In[ ]:



