
# coding: utf-8

# In[11]:

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


# In[12]:

train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')


# In[13]:

cols = list(train.columns)
cols.remove('target')


# In[14]:

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


# In[15]:

X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values


# In[16]:

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


# In[17]:

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state = 10)


# In[ ]:

model = Sequential([
    Dense(units=1024, kernel_initializer='uniform', input_dim=5, activation='relu'),
    Dense(units=512, kernel_initializer='uniform', activation='relu'),
    Dropout(0.25),
    Dense(128, kernel_initializer='uniform', activation='relu'),
    Dense(64, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=20)


# In[ ]:



