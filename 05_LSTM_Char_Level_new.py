#!/usr/bin/env python
# coding: utf-8

# In[14]:


from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io, os
import pickle
import string
from keras.utils import np_utils


# In[2]:


dialogues_dict = pickle.load(open('dialogues.pkl', 'rb'))


# In[3]:


harry_dialogue = dialogues_dict['HARRY'].copy()


# In[1]:


# harry_dialogue


# In[5]:


harry_corpus = ' '.join(harry_dialogue)


# In[2]:


# harry_corpus


# In[7]:


print('corpus length:', len(harry_corpus))


# In[8]:


# create mapping of unique chars to integers
chars = sorted(list(set(harry_corpus)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


# In[11]:


n_chars = len(harry_corpus)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


# In[12]:


# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = harry_corpus[i:i + seq_length]
    seq_out = harry_corpus[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# In[15]:


# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# In[16]:

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# In[17]:


# define the checkpoint
filepath="lstm5-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [es, checkpoint]


# In[18]:

model.fit(X, y, epochs=300, batch_size=128, callbacks=callbacks_list)


# In[19]:






