#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("quote_dataset.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


quotes = df['quote']
quotes.head()


# In[6]:


quotes = quotes.str.lower()


# In[7]:


import string
translator = str.maketrans('', '', string.punctuation)
quotes = quotes.apply(lambda x: x.translate(translator))


# In[8]:


quotes.head()


# In[9]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[10]:


vocab_size = 8978

tokinizer = Tokenizer(num_words=vocab_size)
tokinizer.fit_on_texts(quotes)


# In[11]:


word_index = tokinizer.word_index
print(len(word_index))
list(word_index.items())[:10]


# In[12]:


sequence = tokinizer.texts_to_sequences(quotes)


# In[13]:


for i in range(3):
  print(quotes[i])


# In[14]:


for i in range(3):
  print(sequence[i])


# In[15]:


X = []
y = []

for seq in sequence:
  for i in range(1,len(seq)):
    input_seq = seq[:i]
    output_seq = seq[i]
    X.append(input_seq)
    y.append(output_seq)


# In[16]:


X


# In[17]:


max_len = max(len(x) for x in X)
print(max_len)


# In[18]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
X_padded = pad_sequences(X, maxlen=max_len, padding='pre')


# In[19]:


y = np.array(y)


# In[20]:


X_padded.shape


# In[21]:


from tensorflow.keras.utils import to_categorical
y_one_hot = to_categorical(y, num_classes=vocab_size)


# In[22]:


y.shape


# In[23]:


y_one_hot.shape


# In[24]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense


# In[25]:


embedding_dim = 50
rnn_units = 128


# In[26]:


rnn_model = Sequential()

rnn_model.add(
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)
)
rnn_model.add(SimpleRNN(units=rnn_units))
rnn_model.add(Dense(units=vocab_size, activation='softmax'))


# In[27]:


rnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[28]:


rnn_model.summary()


# In[29]:


lstm_model = Sequential()
lstm_model.add(
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)
)
lstm_model.add(LSTM(units=rnn_units))
lstm_model.add(Dense(units=vocab_size, activation='softmax'))


# In[30]:


lstm_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[31]:


lstm_model.summary()


# In[32]:


epochs = 10
batch_size = 128


# In[33]:


history_rnn = rnn_model.fit(
    X_padded,
    y_one_hot,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1
)


# In[34]:


epochs = 100
batch_size = 128

history_lstm = lstm_model.fit(
    X_padded,
    y_one_hot,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1
)


# In[35]:


lstm_model.save('lstm_model.h5')


# In[36]:


index_to_word = {}
for word, index in word_index.items():
  index_to_word[index] = word


# In[38]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[39]:


def predictor(model,tokenizer,text,max_len):
  text = text.lower()

  seq = tokenizer.texts_to_sequences([text])[0]
  seq = pad_sequences([seq], maxlen=max_len, padding='pre')

  pred = model.predict(seq,verbose = 0)
  pred_index = np.argmax(pred)
  return index_to_word[pred_index]


# In[56]:


seed_text = "it is our"
next_word = predictor(lstm_model,tokinizer,seed_text,max_len)
print(next_word)


# In[65]:


def generate_text(model,tokenizer,seed_text,max_len,n_words):
  for _ in range(n_words):
    next_word = predictor(model,tokenizer,seed_text,max_len)
    if next_word == "":
      break
    seed_text += " " + next_word
  return seed_text

seed = "it is our "
generated_text = generate_text(lstm_model,tokinizer,seed,max_len,10)
print(generated_text)


# In[46]:


import pickle
with open("tokenizer.pkl", "wb") as f:
  pickle.dump(tokinizer, f)


# In[47]:


with open("max_len.pkl", "wb") as f:
  pickle.dump(max_len, f)


# In[ ]:




