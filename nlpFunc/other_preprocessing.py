# tokenizers/other_preprocessing.py
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_punctuation_with_whiteSpace(text):
    return re.sub('[%s]' % re.escape(string.punctuation),' ',text)
    
def remove_digits(text):
    return re.sub('\w*\d\w*', ' ',text)

def remove_stopWords(text):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text)
    return vectorizer.get_feature_names_out()


def expand_user_query(query):
    """Expands the user query by finding synonyms for meaningful content words only."""
    # Helper function to get synonyms (embedded within main function)
    def get_synonyms(word):
        synonyms = set()
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != word:
                    synonyms.add(name)
        return synonyms
    
    # Use NLTK's built-in stopwords
    stop_words = set(stopwords.words('english'))
    
    # Main expansion logic
    tokens = word_tokenize(query.lower())
    expanded_terms = set(tokens)  # Start with the original words
    
    # Collect synonym expansions
    synonym_expansions = []
    
    for word in tokens:
        if word not in stop_words:  # Only expand non-stopwords
            synonyms = get_synonyms(word)
            if synonyms:
                expansion_line = f" - {word}: {', '.join(sorted(synonyms))}"
                synonym_expansions.append(expansion_line)
                expanded_terms.update(synonyms)
    
    # Create result object
    result = {
        'synonym_expansions': '\n'.join(synonym_expansions),
        'expanded_query': " OR ".join(sorted(expanded_terms)),
        'original_query': query
    }
    return result