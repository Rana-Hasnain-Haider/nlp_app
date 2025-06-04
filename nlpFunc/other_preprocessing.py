# tokenizers/other_preprocessing.py
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk
from nltk.tree import Tree

def word_token(text):
    return word_tokenize(text)
    
def remove_punctuation_with_whiteSpace(text):
    return re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    
def remove_digits(text):
    return re.sub('\w*\d\w*', ' ',text)

def remove_stopWords(text):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text)
    return vectorizer.get_feature_names_out()

def Lancaster_stemmer(text):
    tokens = word_token(text)
    stemmer = LancasterStemmer()
    stemm = lambda x:stemmer.stem(x)
    res = list(map(stemm,tokens))
    return res

def porter_stemmer(text):
    stemmer = PorterStemmer()
    tokens = word_token(text)
    stems = [stemmer.stem(word) for word in tokens]
    return stems

def pos_tagging(text):
    tokens = word_token(text)
    tag_tokens = pos_tag(tokens)
    return tag_tokens

def NER(text):
    tokens = word_token(text)
    tagged = pos_tag(tokens)
    ner_tree = ne_chunk(tagged)
    named_entities = []
    for subtree in ner_tree:
        if isinstance(subtree, Tree):  # This means it's a named entity
            entity = " ".join([token for token, pos in subtree.leaves()])
            label = subtree.label()
            named_entities.append((entity, label))

   
    # print("Named Entities:")
    # for entity, label in named_entities:
    #     print(f"{entity}: {label}")
    return named_entities
