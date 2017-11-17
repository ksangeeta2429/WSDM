
# coding: utf-8

# In[ ]:

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


# In[ ]:

song_cols = ['song_id', 'genre_ids', 'song_length', 'language']
songs = pd.read_csv('../Data/songs.csv', usecols = song_cols)

headers = ['song_id', 'translated_names']
artists = pd.read_csv('../New_Data/tr_artists.csv', usecols = headers)
composers = pd.read_csv('../New_Data/tr_composer.csv', usecols = headers)
lyricists = pd.read_csv('../New_Data/tr_lyricists.csv', usecols = headers)
members = pd.read_csv('../Data/members.csv')

members = members.drop(['bd', 'gender','registration_init_time','expiration_date'], axis=1)


# In[ ]:

train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')


# In[ ]:

songs_new = songs.merge(artists, on='song_id', how='left')
songs_new = songs_new.merge(lyricists, on='song_id', how='left')
songs_new = songs_new.merge(composers, on='song_id', how='left')


# In[ ]:

train = train.merge(songs_new, on='song_id', how='left')
test = test.merge(songs_new, on='song_id', how='left')

#members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
#members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
#members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

#members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
#members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
#members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
#members = members.drop(['registration_init_time'], axis=1)

train = train.merge(members, on='msno', how='left')
test = test.merge(members, on='msno', how='left')

train = train.fillna(-2)
test = test.fillna(-2)


# In[ ]:

cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])


# In[ ]:

X = np.array(train.drop(['target'], axis=1))
Y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
Y_test = test['id'].values


# In[ ]:

cols_name = list(train.columns)
X_new = X

max_vals = X_new.max(axis = 0).transpose()
min_vals = X_new.min(axis = 0).transpose()
mean_vals = np.mean(X_new, axis = 0).transpose()

max_vals_2 = X_test.max(axis = 0).transpose()
min_vals_2 = X_test.min(axis = 0).transpose()
mean_vals_2 = np.mean(X_test, axis = 0).transpose()

#training set
X_new = X_new - mean_vals
X_new = X_new / (max_vals - min_vals)
X_new = np.around(X_new,decimals = 2)

#testing set
X_new_test = X_test - mean_vals
X_new_test = X_new_test / (max_vals - min_vals)
X_new_test = np.around(X_new_test,decimals = 2)
#print X_new_test.max(axis = 0)
#print X_new_test.min(axis = 0)


# In[ ]:

#np.savetxt("train.csv", X_new, delimiter=",")
#np.savetxt("test.csv", X_new_test, delimiter=",")


# In[ ]:

X_train, X_val, Y_train, Y_val = train_test_split(X_new, Y, test_size=0.1, random_state = 10)
    
del X; 
del members, songs, artists, composers, lyricists;
del train, test;
gc.collect();


# In[ ]:

model = Sequential([
    Dense(units=1024, kernel_initializer='uniform', input_dim=X_train.shape[1], activation='relu'),
    Dense(units=512, kernel_initializer='uniform', activation='relu'),
    Dropout(0.25),
    Dense(128, kernel_initializer='uniform', activation='relu'),
    Dense(64, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=128, epochs=20)


# In[ ]:

model.save('../Models/dnn_preprocessed.h5')


# In[ ]:

predicted = model.predict(X_new_test, batch_size=128, verbose=0)


# In[ ]:

headers = ['id', 'predicted']
df_new = pd.DataFrame(columns=headers)
df_new['id'] = Y_test
df_new['predicted'] = predicted
df_new.to_csv('test.csv', index=False)

