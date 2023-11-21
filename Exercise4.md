# EXERCISE 4

## OBJECTIVE

Implement application for Word2Vec for NLP Using Python on "email" dataset
(which contains attributes like category(whether the category is ham or spam) and message(Text message)).
On this application you have to perform following operations
1. generate Embeddings
2. visualize embeddings
3. Cleaning the data
4. creating a Corpus and vectors
5. Visualize email word vectors
6. Analyzing and predicting using word embeddings

## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1. Load email dataset
2. Generating Embeddings and clean the data
	    3. Explore the Embeddings
	    4. analyzing and predicting
	    5. Predicting Embedding words
## DESCRIPTION / PROCEDURE
Download the data set
https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
```python
from gensim.models import word2vec, FastText
import pandas as pd
import re

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import plotly.graph_objects as go

import numpy as np

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('emails.csv')


# Sample sentences
sentences = [['i', 'like', 'apple', 'pie', 'for', 'dessert'],
           ['i', 'dont', 'drive', 'fast', 'cars'],
           ['data', 'science', 'is', 'fun'],
           ['chocolate', 'is', 'my', 'favorite'],
           ['my', 'favorite', 'movie', 'is', 'predator']]
# Generate Embeddings

# train word2vec model
w2v = word2vec(sentences, min_count=1, size = 5)

print(w2v)
#word2vec(vocab=19, size=5, alpha=0.025)
# access vector for one word
print(w2v['chocolate'])

#[-0.04609262 -0.04943436 -0.08968851 -0.08428907  0.01970964]

#list the vocabulary words
words = list(w2v.wv.vocab)

print(words)

#or show the dictionary of vocab words
w2v.wv.vocab
# Visualize Embeddings

X = w2v[w2v.wv.vocab]
pca = PCA(n_components=2)

result = pca.fit_transform(X)

# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(w2v.wv.vocab)

for i, word in enumerate(words):
   plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()

# Visualizing Email Word Embeddings
df.head()

# Cleaning the Data

clean_txt = []
for w in range(len(df.text)):
   desc = df['text'][w].lower()

   #remove punctuation
   desc = re.sub('[^a-zA-Z]', ' ', desc)

   #remove tags
   desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)

   #remove digits and special chars
   desc=re.sub("(\\d|\\W)+"," ",desc)
   clean_txt.append(desc)

df['clean'] = clean_txt
df.head()

# Creating a Corpus and Vectors
corpus = []
for col in df.clean:
   word_list = col.split(" ")
   corpus.append(word_list)

#show first value
corpus[0:1]

#generate vectors from corpus
model = word2vec(corpus, min_count=1, size = 56)
# Visualizing Email Word Vectors
#pass the embeddings to PCA
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

#create df from the pca results
pca_df = pd.DataFrame(result, columns = ['x','y'])

#add the words for the hover effect
pca_df['word'] = words
pca_df.head()

N = 1000000
words = list(model.wv.vocab)
fig = go.Figure(data=go.Scattergl(
   x = pca_df['x'],
   y = pca_df['y'],
   mode='markers',
   marker=dict(
       color=np.random.randn(N),
       colorscale='Viridis',
       line_width=1
   ),
   text=pca_df['word'],
   textposition="bottom center"
))

fig.show()

# Analyzing and Predicting Using Word Embeddings
#explore embeddings using cosine similarity
model.wv.most_similar('eric')

model.wv.most_similar_cosmul(positive = ['phone', 'number'], negative = ['call'])

model.wv.doesnt_match("phone number prison cell".split())

#save embeddings
file = 'email_embd.txt'
model.wv.save_word2vec_format(filename, binary = False)
```
Output:

 

 

 


