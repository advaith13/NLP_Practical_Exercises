# EXERCISE 3

## OBJECTIVE
Download Wikipedia's page on open source and convert the text to its native forms. 
Try it with various stemming and lemmatizing modules. 
Use Python's timer module to measure their performance.
## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1. Load Wikipedia page as open source
2. Convert the text into native forms
	    3. Identify stemming and lemmatizing modules
	    4.Identify the performance
## DESCRIPTION / PROCEDURE
```python
#install wikipedia module
!pip3 install wikipedia
!pip3 install gtts
!pip3 install playsound
!pip3 install nltk
!python -m nltk.downloader popular
import wikipedia
#To get list of open source text from wikipedia page
result = wikipedia.search("open source")
# get the page: Neural network
page = wikipedia.page(result[0])
# get the whole wikipedia page text (content)
content = page.content
print(type(content))
import gtts
from playsound import playsound
# make request to google to get synthesis

tts = gtts.gTTS(content)
# save the audio file
tts.save("content.mp3")
import nltk

from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
text=page.content
#Sentence tokenization:
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
print(len(sentences), 'sentences:', sentences)
#Word tokenization:
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print("Original words")
print(len(tokens), 'words:', tokens)
#Creating object using PorterStemmer
stemmer = PorterStemmer()
for token in tokens:
    print(token + ' --> ' + stemmer.stem(token))
#perform Lemmatization suing spacy module
import spacy as sp
import spacy
#Get on sentence from list of sentences to find lemma
sp = spacy.load('en_core_web_sm')
s5=sp(sentences[4])
print(s5)
for word in s5:
    print(word.text + '  ===>', word.lemma_)
#Python's timer module to measure their performance

import time
start_time = time.time()
print("Time elapsed after some level wait...")
print("The start time is", start_time)
time.sleep(1)
end_time= time.time()
print("The end time is", end_time)
```
## Output:
```
Stemming:

2008 --> 2008
) --> )
. --> .
`` --> ``
The --> the
Economic --> econom
Properties --> properti
of --> of
Software --> softwar
'' --> ''
( --> (
PDF --> pdf
) --> )
. --> .
Jena --> jena
Economic --> econom
Research --> research
Papers --> paper
……….

Lemmatization:

The open-source movement in software began as a response to the limitations of proprietary code.
The  ===> the
open  ===> open
-  ===> -
source  ===> source
movement  ===> movement
in  ===> in
software  ===> software
began  ===> begin
as  ===> as
a  ===> a
response  ===> response
to  ===> to
the  ===> the
limitations  ===> limitation
of  ===> of
proprietary  ===> proprietary
code  ===> code
.  ===> .
________________________________________
Time elapsed after some level wait...
The start time is 1658606683.5451832
The end time is 1658606684.5493498
```

