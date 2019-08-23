from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io, os
import pickle
import string
from keras.utils import np_utils



# load the network weights
filename = "weights-improvement-20-2.4384.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[22]:


int_to_char = dict((i, c) for i, c in enumerate(chars))


# In[23]:


# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print( "Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")


# In[ ]: