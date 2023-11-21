
# EXPERIMENT 7
## OBJECTIVE
Create a RNN model to predict if the movie review is positive or negative. First load the “IMDB movie review” dataset. This dataset has 50k reviews of different movies. It is a benchmark dataset used in text-classification to train and test the Machine Learning and Deep Learning model.
Perform following operations to build the RNN model for text classification
Preprocessing the Data
Input and output label selection
Build the model
Explore Vanishing Gradients
Model compilation
## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1. Load “IMDB movie review” dataset
2. Process the data
	    3. Build the model
	    4. Compile the model
	    5. Explore Vanishing Gradients
	    6. model compilation
## DESCRIPTION / PROCEDURE

Load the dataset click on : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Build RNN model:
 Recurrent Neural Network
Source Code:
```python
from tensorflow.keras.datasets import imdb
from google.colab import drive
drive.mount("/content/drive")
path = 'drive/My Drive/Colab Notebooks/IMDB Dataset.csv'
from keras.preprocessing.text import one_hot
sentence=['Fast cars are good',
          'Football is a famous sport',
          'Be happy Be positive']
vocab_size=1000
encoded_docs=[one_hot(d,vocab_size) for d in sentence]
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
import numpy as np
embedding_length=5
max_length=10
encoded_docs=pad_sequences(encoded_docs,truncating='post',padding='post',maxlen=max_length)
print(encoded_docs)
model=Sequential()
model.add(Embedding(vocab_size,embedding_length,input_length=max_length))
model.compile('rmsprop','mse')
model.summary()
output=model.predict(encoded_docs)
print(output.shape)
print(output)
import numpy as np
import tensorflow as tf
"""Loading the Dataset"""
from tensorflow.keras.datasets import imdb
"""### **Data Preprocessing**"""
words=20000
max_length=100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=words)
"""Padding the Text"""
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
word_size=words
word_size
embed_size=128
"""### Building a Recurrent Neural Network"""
imdb_model=tf.keras.Sequential()
# Embedding Layer
imdb_model.add(tf.keras.layers.Embedding(word_size, embed_size, input_shape=(x_train.shape[1],)))
# LSTM Layer
imdb_model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
# Output Layer
imdb_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
imdb_model.summary()
"""#### Compiling the model"""
imdb_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
"""#### Training the model"""
imdb_model.fit(x_train, y_train, epochs=5, batch_size=128)
test_loss, test_acurracy = imdb_model.evaluate(x_test, y_test)
print("Test accuracy: {}".format(test_acurracy))
```
Executed Code:
https://colab.research.google.com/drive/1umg4THVNKrBU9XpV3S6nkfDlsNFm8j7G#scrollTo=kM1xQRT6nm0N


Output:
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, 100, 128)          2560000   
                                                                 
 lstm (LSTM)                 (None, 128)               131584    
                                                                 
 dense (Dense)               (None, 1)                 129       
                                                                 
=================================================================
Total params: 2,691,713
Trainable params: 2,691,713
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5
196/196 [==============================] - 79s 391ms/step - loss: 0.4665 - accuracy: 0.7866
Epoch 2/5
196/196 [==============================] - 76s 388ms/step - loss: 0.2941 - accuracy: 0.8826
Epoch 3/5
196/196 [==============================] - 76s 389ms/step - loss: 0.2292 - accuracy: 0.9128
Epoch 4/5
196/196 [==============================] - 76s 388ms/step - loss: 0.1927 - accuracy: 0.9294
Epoch 5/5
196/196 [==============================] - 77s 391ms/step - loss: 0.1578 - accuracy: 0.9418
782/782 [==============================] - 28s 35ms/step - loss: 0.5458 - accuracy: 0.7984
Test accuracy: 0.7983599901199341
```


