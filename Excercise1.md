# EXERCISE1

## OBJECTIVE
1.	Create a basic NLP program to find words, phrases, names and concepts using "spacy.blank" to create the English nlp object. Process the text and instantiate a Doc object in the variable doc. Select the first token of the Doc and print its text.
## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1.	Install Required python modules like spacy and load English words
2.	Extract the words/tokens from the sentence
3.	Process the text
4.	Identify the first token in the document
## DESCRIPTION / PROCEDURE
```jupyter
# Import the spacy library
import spacy
# Install English based model to find words,phrases,names
!python -m spacy download en_core_web_sm
#load the en_core_web_sm to find phrases
nlp = spacy.load('en_core_web_sm')
text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
# Import the spacy library
nlp = spacy.blank("en")
# Process the text
doc = nlp(text)
# Print the document text
print(doc.text)
# remember to restart the runtime 
# (do not 'Reset all runtimes'!)
!python -m spacy download de_core_news_sm
# Load the 'de_core_news_sm' model – spaCy imported
import spacy
nlp = spacy.load('de_core_news_sm')
text = "Be Indian and Buy Indian"
# Process the text
doc = nlp(text)
# Print the document text
print(doc.text)
text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
# Process the text
doc = nlp(text)
for token in doc:
    # Get the token text, part-of-speech tag and dependency label
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    # This is for formatting only
    print('{:<12}{:<10}{:<10}'.format(token_text, token_pos, token_dep))
text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"
# Process the text
doc = nlp(text)
for token in doc:
    # Get the token text, part-of-speech tag and dependency label
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    # This is for formatting only
    print('{:<12}{:<10}{:<10}'.format(token_text, token_pos, token_dep))
first_token=doc[0]
print("First Toekn: ",first_token)

text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"

# Process the text
doc = nlp(text)

print("Text lables:")
for ent in doc.ents:
    # print the entity text and its label
    print(ent.text, ent.label_)

text = "New iPhone X release date leaked as Apple reveals pre-orders by mistake"

# Process the text
doc = nlp(text)

# Iterate over the entities
for ent in doc.ents:
    # print the entity text and label
    print(ent.text, ent.label_)
spacy.explain('MISC')
```

## OUTPUT:
```
Document text:
It’s official: Apple is the first U.S. public company to reach a $1 trillion market value

Text after Reset:
Be Indian and Buy Indian
Word,position
It          PROPN     pnc       
’s          PROPN     ROOT      
official    ADV       mnr       
:           PUNCT     punct     
Apple       X         pnc       
is          X         uc        
the         X         uc        
first       X         pnc       
U.S.        X         app       
public      X         punct     
company     X         uc        
to          X         uc        
reach       X         uc        
a           PROPN     pnc       
$           PROPN     ag        
1           NUM       pnc       
trillion    NOUN      ROOT      
market      PROPN     pnc       
value       PROPN     nk        
First Toekn:  It

Text lables:
It’s official: Apple is the first U.S. public company to reach MISC
market value MISC
New iPhone X release date leaked MISC
pre-orders MISC
Miscellaneous entities, e.g. events, nationalities, products or works of art
```

