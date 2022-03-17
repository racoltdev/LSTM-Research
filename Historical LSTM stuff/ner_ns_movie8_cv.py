# -*- coding: utf-8 -*-
"""NER_ns_movie8_CV.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n3_4TeyUeqWdSQJdylkkqCzYJFQQqnH-
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
print('GPU detected:', tf.config.list_physical_devices('GPU'))

tf.compat.v1.enable_eager_execution()

"""### Load data in Colab"""

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("MITMovie_dataset.csv", encoding="utf-8")
df.head(20)

"""###Split dataset into individual sentences"""

# inserting the sentence indicator
# detect an empty space and increse sentence count 

Sentnum = 1
#d = {'Sentence #':[], 'Word':[], 'Tag':[]}
data = pd.DataFrame(columns=['Sentence #','Word','Tag'])
print(data)

for ind, row in df.iterrows():
  #print(row['Word'], row['NER'])
 
  if row.notna().any():
    #d2  = {'Sentence #':Sentnum, 'Word':row['Word'], 'Tag':row['NER']}
    dat = [[Sentnum,row['Word'],row['NER']]]
    #print(dat)
    df2 = pd.DataFrame(dat, columns=['Sentence #','Word','Tag'])
    #print(df2)
    data = data.append(df2)  
    #print(data)
  else:
    Sentnum +=1
    #print(Sentnum)

data.head(20)

# Find the number of unique words and unique tags in dataset
# Will use it to encode the sentences

print("Unique words in dataset:", data['Word'].nunique())
print("Unique tags in dataset:", data['Tag'].nunique())

# Make a list of all words and use the set method to have unique entries 
# Add the flag ENDFILL to the end of the word set
# The ENDFILL flag will be used to pad short entences
words = list(set(data["Word"].values))
words.append("ENDFILL")
words_count = len(words)

# Make a list of all tags and use the set method to have unique entries 
tags = list(set(data["Tag"].values))
tags_count = len(tags)

"""### Obtain individual sentences with words and tags"""

# this method retrieves individual sentences
# zip() functions joins tuples together in one big tuple
# lambda funtion returns the variable s
# the groupby() function groups the input data by sentence numbers
# the apply(agg_func) method applies agg_func to grouped sentences
# method get_next loops through the data until all sentencess are processed

class ObtainSentence(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

get_sentence = ObtainSentence(data)
sentences = get_sentence.sentences

#Check the result by inspecting the first sentence
sentences[0]

"""### Create dictionaries for words and tags"""

# We map all unique words into indexes from 1 to words_count
# the same for tags
#words_index = {w: i + 1 for i, w in enumerate(words)} 
words_index = {w: i for i, w in enumerate(words)} 
tags_index = {t: i for i, t in enumerate(tags)}

tags_index

"""### Make all input sentences the same length with padding"""

#creates an array of sentences lengths and plots it in 50 bin histogram
# this histogram is used to define optimum sentence length
plt.hist([len(s) for s in sentences], bins=35)
plt.show()

# now we will pad sequences shprter than max_sentence_length
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_sentence_length = 50

X = [[words_index[w[0]] for w in s] for s in sentences] #convert sentences into numeric sequences using word2idx
X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=words_index["ENDFILL"]) #keras function padding the sequences

y = [[tags_index[w[1]] for w in s] for s in sentences]  #convert tags into numeric sequences using word2idx
y = pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=tags_index["O"]) #padding the tags sequences

"""### Split data into the training and test sets"""

from sklearn.model_selection import LeaveOneOut,KFold

def make_dataset(X_data,y_data,n_splits):

    def gen():
        for train_index, test_index in KFold(n_splits).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train,y_train,X_test,y_test

    return tf.data.Dataset.from_generator(gen, (tf.float64,tf.float64,tf.float64,tf.float64))

# prepare K-fold split for cross-validation
dataset=make_dataset(X,y,5)

foldcount = 1
for X_train,y_train,X_test,y_test in dataset:
  print(foldcount)
  #print(X_train[5])
  #print(X_train.shape)
  #print(X_test.shape)
  #print(y_train.shape)
  #print(y_test.shape)
  foldcount +=1
  #print(X_test[5])

"""### Define a bidirectional LSTM model"""

!pip install tensorflow-addons

from tensorflow import keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, GRU
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
import tensorflow_addons as tfa
from sklearn import metrics

#fwd_layer = tf.keras.layers.RNN(keras.layers.LSTMCell(128, recurrent_dropout=0.1), return_sequences=True)
#bwd_layer = tf.keras.layers.RNN(keras.layers.LSTMCell(128, recurrent_dropout=0.1), return_sequences=True, go_backwards=True)

#fwd_layer = tf.keras.layers.RNN(tf.keras.layers.GRUCell(512, recurrent_dropout=0.1), return_sequences=True)
#bwd_layer = tf.keras.layers.RNN(tf.keras.layers.GRUCell(512, recurrent_dropout=0.1), return_sequences=True, go_backwards=True)

#fwd_layer = tf.keras.layers.RNN(tfa.rnn.PeepholeLSTMCell(128, recurrent_dropout=0.1), return_sequences=True)
#bwd_layer = tf.keras.layers.RNN(tfa.rnn.PeepholeLSTMCell(128, recurrent_dropout=0.1), return_sequences=True, go_backwards=True)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=words_count, output_dim=50, input_length=max_sentence_length),
    tf.keras.layers.SpatialDropout1D(0.1),
    #tf.keras.layers.RNN(keras.layers.LSTMCell(128, recurrent_dropout=0.1), return_sequences=True),
    #tf.keras.layers.RNN(keras.layers.LSTMCell(128, recurrent_dropout=0.1), return_sequences=True),
    #tf.keras.layers.RNN(tfa.rnn.PeepholeLSTMCell(128, recurrent_dropout=0.1), return_sequences=True),
    #tf.keras.layers.RNN(tfa.rnn.PeepholeLSTMCell(128, recurrent_dropout=0.1), return_sequences=True),
    
    #tf.keras.layers.Bidirectional(layer = fwd_layer, backward_layer= bwd_layer),
    #tf.keras.layers.Bidirectional(layer = fwd_layer, backward_layer= bwd_layer),

    #tf.keras.layers.Bidirectional(tf.keras.layers.RNN(keras.layers.LSTMCell(128, recurrent_dropout=0.1), return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.RNN(keras.layers.LSTMCell(128, recurrent_dropout=0.1), return_sequences=True)),

    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, recurrent_dropout=0.1)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, recurrent_dropout=0.1)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),

    tf.keras.layers.TimeDistributed(keras.layers.Dense(tags_count, activation="softmax"))
])
model.summary()

"""### Compile the model"""

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
              )

"""### Load the Tensorboard"""

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard
import datetime

# Clear any logs from previous runs
!rm -rf ./logs/

#define directory for new Tensorboard logs
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

"""### Train the model"""

#train the model using .fit method
fold = 0
lss = []
acc = []
acc_nofill = []
sls = 0
sac = 0
sacnofill = 0
for X_train,y_train,X_test,y_test in dataset:
  print('Fold = ',fold+1)
  fold +=1

  # reset the current model
  #tf.keras.backend.clear_session()

  del model

  model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=words_count, output_dim=50, input_length=max_sentence_length),
    tf.keras.layers.SpatialDropout1D(0.1),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, recurrent_dropout=0.1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, recurrent_dropout=0.1)),
    tf.keras.layers.TimeDistributed(keras.layers.Dense(tags_count, activation="softmax"))
  ])

  model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
              )

  history = model.fit(
      x=X_train,
      y=y_train,
      validation_data=(X_test,y_test),
      batch_size=32, 
      epochs = 10,
      callbacks=[tensorboard_callback],
      verbose=1
  )

  # Calculate accuracy without accounting for padded values:
  truetags = []
  predictedtags =[]
  numsentences = 0
  accuracy = 0
  print(X_test.shape[0])
  for i in range(X_test.shape[0]):
    current_accuracy = 0 
    count = 0
    correct = 0
    p = model.predict(np.array([X_test[i]]))
    p = np.argmax(p, axis=-1)
    y_true = y_test[i]
    
    len1 = len(X_test[i].numpy())
    #print('Test set sentences =',len1)
    for j in range(len1):
      w = int(X_test[i,j].numpy())
      true = int(y_true[j].numpy())
      pred = p[0,j]
      if words[w] != "ENDFILL":
        count +=1
        truetags.append(tags[true])
        predictedtags.append(tags[pred])
        if tags[true] == tags[pred]:
          correct +=1
      
    current_accuracy = correct/count
    #print(current_accuracy)
    accuracy = accuracy + current_accuracy
    numsentences +=1

  print('Number of test sentences = ', numsentences)
  #print('Number of words w/o ENDFILL = ', count')
  accuracy = accuracy/numsentences
  acc_nofill.append(accuracy)  
  print('Accuracy on test set = ', acc_nofill)
  print('Length of true tags  = ', len(truetags))
  print('Length of predicted tags = ', len(predictedtags))
  print(metrics.classification_report(truetags,predictedtags))

  [ls,ac] = model.evaluate(X_test, y_test)
  lss.append(ls)
  acc.append(ac)
  print('TF loss = ',ls)
  print(' TF accuracy =', acc)

acc_cv = sum(acc)/len(acc)
lss_cv = sum(lss)/len(lss)
acc_cv_nofill = sum(acc_nofill)/len(acc_nofill)

## let's compute standard error for accuracy and loss
for i in range(fold):
  sls = sls +(lss[i]-lss_cv)**2
  sac = sac +(acc[i]-acc_cv)**2
  sacnofill = sacnofill +(acc_nofill[i]-acc_cv_nofill)**2

n=len(acc)
se_lss = (sls/(n-1))**0.5/(n)**0.5
se_acc = (sac/(n-1))**0.5/(n)**0.5
se_acc_nf = (sacnofill/(n-1))**0.5/(n)**0.5

print('TF CV accuracy = ', '{:06.4f}'.format(acc_cv),' +/- ', '{:06.4f}'.format(se_acc))
print('TF CV loss = ', '{:06.4f}'.format(lss_cv),' +/- ', '{:06.4f}'.format(se_lss))
print('CV accuracy w/o endfill = ', '{:06.4f}'.format(acc_cv_nofill),' +/- ', '{:06.4f}'.format(se_acc_nf))

#print('N = ',n)
#print('TF CV standard error = ', se_acc)
##print('TF loss standard error = ', se_lss)
#print('CV standard error w/o endfill= ', se_acc_nf)

[lss,acc] = model.evaluate(X_test, y_test)
print(acc, lss)

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs/fita

"""### Evaluate model performance"""

i = np.random.randint(0, X_test.shape[0]) 
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]

print("{:15}{:8}\t {}\n".format("Word", "True_Tag", "Predicted_Tag"))
print("-" *50)

len1 = len(X_test[i].numpy())
for j in range(len1):
  w = int(X_test[i,j].numpy())
  true = int(y_true[j].numpy())
  predicted = p[0,j]
  print("{:15}{}\t{}".format(words[w], tags[true], tags[predicted]))

#Calculate accuracy without accounting for padded values:
truetags = []
predictedtags =[]
numsentences = 0
accuracy = 0
for i in range(x_test.shape[0]):
  current_accuracy = 0 
  p = model.predict(np.array([x_test[i]]))
  p = np.argmax(p, axis=-1)
  y_true = y_test[i]
  count = 0
  correct = 0
  for w, true, pred in zip(x_test[i], y_true, p[0]):
    if words[w] != "ENDFILL":
      count +=1
      truetags.append(tags[true])
      predictedtags.append(tags[pred])
      if tags[true] == tags[pred]:
        correct +=1
      
  current_accuracy = correct/count
  accuracy = accuracy + current_accuracy
  numsentences +=1

print('Number of test sentences = ', numsentences)
accuracy = accuracy/numsentences  
print('Accuracy on test set = ', accuracy)
print('Length of true tags  = ', len(truetags))
print('Length of predicted tags = ', len(predictedtags))

"""###Determine other accuracy metrics and F1 score"""

from sklearn import metrics
print(metrics.classification_report(truetags,predictedtags))