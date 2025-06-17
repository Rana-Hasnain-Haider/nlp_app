from nltk.tokenize import word_tokenize
from nltk.tag import RegexpTagger
import pickle
from nltk.tag import pos_tag
import os

# Get the current file's directory
current_dir = os.path.dirname(__file__)



def rule_based(text):
    tokens = word_tokenize(text)

    # Define rules: (regex pattern, tag)
    patterns = [
        (r'.*ing$', 'VBG'),   # gerunds
        (r'.*ed$', 'VBD'),    # past tense
        (r'.*es$', 'VBZ'),    # 3rd person singular present
        (r'.*ould$', 'MD'),   # modals
        (r'.*\'s$', 'POS'),   # possessive nouns
        (r'.*s$', 'NNS'),     # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # numbers
        (r'^[A-Z].*$', 'NNP'),  # proper nouns
        (r'.*', 'NN')          # default: noun
    ]
    tagger = RegexpTagger(patterns)
    tagged = tagger.tag(tokens)
    return tagged


def HMM_tagger(sentence):
    # Define the same estimator as used during training
    model_path = os.path.join(current_dir, '..', 'taggers', "hmm_pos_tagger.pkl")
    with open(model_path, "rb") as f:
        hmm_tagger = pickle.load(f)

    # Tag a new sentence
    sentence = "The quick brown fox jumps over the lazy dog".split()
    tagged_sentence = hmm_tagger.tag(sentence)
        
    return tagged_sentence

def Maximum_Entropy_tagger(text):
    tokens = word_tokenize(text)
    tag_tokens = pos_tag(tokens)
    return tag_tokens

def brill_tagger(sentence):
    model_path = os.path.join(current_dir, '..', 'taggers', "brill_tagger.pkl")
    with open(model_path, "rb") as f:
        brill_tagger = pickle.load(f)
    
    # Tag the sentence
    tokens = sentence.split()
    tagged_sentence = brill_tagger.tag(tokens)
    
    return tagged_sentence

def Logistic_Regression_tagger(sentence: str):
    # Load model and vectorizer
    model_path = os.path.join(current_dir, '..', 'taggers', "pos_logreg_model.pkl")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    vec_path = os.path.join(current_dir, '..', 'taggers', "pos_vectorizer.pkl")
    with open(vec_path, "rb") as f:
        vec = pickle.load(f)

    words = sentence.split()
    tagged = []

    for i in range(len(words)):
        word = words[i]
        prev_word = words[i - 1] if i > 0 else "<START>"
        next_word = words[i + 1] if i < len(words) - 1 else "<END>"

        features = {
            'word': word,
            'prev_word': prev_word,
            'next_word': next_word,
            'suffix(1)': word[-1:],
            'suffix(2)': word[-2:],
            'prefix(1)': word[:1],
            'prefix(2)': word[:2],
            'is_title': word.istitle(),
            'is_upper': word.isupper(),
            'is_lower': word.islower(),
            'is_digit': word.isdigit()
        }

        # Transform and predict
        X_input = vec.transform([features])
        predicted_tag = clf.predict(X_input)[0]
        tagged.append((word, predicted_tag))

    return tagged
