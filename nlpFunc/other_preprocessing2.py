# tokenizers/other_preprocessing2.py

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk
from nltk.tree import Tree
from nltk.corpus import wordnet as wn


def word_token(text):
    return word_tokenize(text)
    
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
    return named_entities


def lemmatize_sentence(sentence):
    """Lemmatizes a sentence by mapping POS tags and applying lemmatization."""
    # Helper function to map POS tags from Penn Treebank to WordNet (embedded)
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('V'):
            return wn.VERB
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN  # Default to noun
    
    # Main lemmatization logic
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    
    lemmatized_words = []
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        lemmatized_words.append(lemma)
    
    return lemmatized_words