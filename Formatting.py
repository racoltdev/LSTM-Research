#looking at rewriting my code


#Importing the same stuff as the example code (I dont entirely know what I need)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime

df = pd.read_csv("food.csv", encoding="utf-8")


#Words and tags are sets
words = set()
words.add("ENDFILL")
tags = set()
tags.add("O")

#base values for things
sentnum = 0
indnum = 0

#iterate though the words
for ind, row in df.iterrows():
    if row.notna().any():
        words.add(row['Word'])
        tags.add(row['NER'])
    else:
        sentnum+=1
        if sentnum%500 == 0:
            print(sentnum)

#These two lines were from the original example, I do not know of how to do it better
words_index = {w: i for i, w in enumerate(words)} 
tags_index = {t: i for i, t in enumerate(tags)}

#ballpark how big the array needs to be
estWordLength = int(df.size/sentnum)+4
columns, rows = (sentnum+1,estWordLength)

#generate words and tangs arrays
data_w = [[words_index["ENDFILL"] for i in range(rows)] for j in range(columns)]
data_t = [[tags_index["O"] for i in range(rows)] for j in range(columns)]

x = np.array(data_w[0])

print(x)

#reset sentnum
sentnum = 0

#iterate though words again and this time fill in the arrays.
for ind, row in df.iterrows():
    if row.notna().any():
        if indnum<estWordLength:
            data_w[sentnum][indnum] = words_index[row['Word']]
            data_t[sentnum][indnum] = tags_index[row['NER']]
        indnum+=1
    else:
        sentnum+=1
        if sentnum%500 == 0:
            print(sentnum)
        indnum=0

sentnum+=1

#I think at this point the data is set up

#----------------------------------------------------------------No longer my stuff after this line

x_train, x_test, y_train, y_test = train_test_split(data_w, data_t, test_size=0.02, random_state=1)

#I have no clue what this does I am just using what was in the example 
# (I will look further once I have a better handle on things)
input_word = Input(shape=(estWordLength,))
model = Embedding(input_dim=len(words_index), output_dim=50, input_length=estWordLength)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(len(tags_index), activation="softmax"))(model)
model = Model(input_word, out)
model.summary()

#This appears to say what settings to put it on and how it optimizes
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
              )

#this appears to be saying where to save the logs (I have not looked at logs yet)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#I think this tells it to run
history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test,y_test),
    batch_size=32, 
    epochs = 2,
    callbacks=[tensorboard_callback],
    verbose=1
)

# Inspect prediction results for a random sentence
i = np.random.randint(0, len(x_test)-1)
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]

#print the comparison
print("{:15}{:8}\t {}\n".format("Word", "True_Tag", "Predicted_Tag"))
print("-" *50)
for w, true, predicted in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(list(words)[w], list(tags)[true], list(tags)[predicted]))

model.evaluate(x_test, y_test)
