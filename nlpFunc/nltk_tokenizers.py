# tokenizers/nltk_tokenizers.py
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
import re

def word_tokenizer(text):
    return word_tokenize(text)

def sentence_tokenizer(text):
    return sent_tokenize(text)

def generate_ngrams(text, n):
    tokens = word_tokenize(text)
    return list(ngrams(tokens, n))

def cap_tokenizer(text):
    capTokenizer = RegexpTokenizer("[A-Z][a-z]+")
    return capTokenizer.tokenize(text)

def general_regexp_tokenizer(text, regexp):
    try:
        # Test if the regex pattern is valid by compiling it
        re.compile(regexp)
        
        # Create tokenizer and tokenize
        general_tokenizer = RegexpTokenizer(regexp)
        tokens = general_tokenizer.tokenize(text)
        
        return tokens
        
    except re.error as e:
        raise re.error(f"Invalid regular expression pattern '{regexp}': {str(e)}")
    
    except Exception as e:
        # Handle any other unexpected errors
        raise RuntimeError(f"An unexpected error occurred during tokenization: {str(e)}")

