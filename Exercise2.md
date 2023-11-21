# Exercise 2

## OBJECTIVE

Using Python libraries, download Wikipedia's page on open source and tokenize the text. And also remove the stopwords. What percentage of the page is stopwords?
## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1. Load Wikipedia page as open source
2. Tokenize the text
3. Remove the stopwords
4.Find the percentage of stopwords
## DESCRIPTION / PROCEDURE
```jupyter
#installl Required module
!pip3 install Wikipedia
!pip3 install nltk
!python -m nltk.downloader popular

import wikipedia
# print the summary of what python is
print(wikipedia.summary("Python Programming Language",sentences=2))
result = wikipedia.search("open source")
# get the page: Neural network
page = wikipedia.page(result[0])

# get the title of the page
title = page.title
# get the categories of the page
categories = page.categories
# get the whole wikipedia page text (content)
content = page.content
print(content)
#to find the page content type
from nltk.tokenize import sent_tokenize
print(type(content))
text=page.content
#Sentence tokenization:
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
print(len(sentences), 'sentences:', sentences)

#Word tokenization:
from nltk.tokenize import word_tokenize
words = word_tokenize(text)
print("Original words")
print(len(words), 'words:', words)

#Find stop words using nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(len(stop_words), "stopwords:", stop_words)



#Find words without stop words
words = [word for word in words if word not in stop_words]
print(len(words), "without stopwords:", words)

#Find words without stop words using spacy module
import spacy
sp = spacy.load('en_core_web_sm')

all_stopwords = sp.Defaults.stop_words
text=page.content
text_tokens = word_tokenize(text)
tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]
print("Tokens without stopwords")
print(len(tokens_without_sw))

#Percentage of stopwords
print(len(all_stopwords),all_stopwords)
print("percentage of the stopwords")
print((len(original_words)-len(ws_words))/len(original_words)*100)
```
## Output:
```
Original words
10174 words

7441 without stopwords
Total stop words 
326

percentage of the stopwords
26.862590918026342
```



