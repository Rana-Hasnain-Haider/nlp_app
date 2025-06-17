from collections import Counter
from collections import defaultdict, Counter
import random

def tokenize(corpus):
    return corpus.lower().split()

def build_unigram_model(corpus):
    tokens = tokenize(corpus)
    total_tokens = len(tokens)
    word_counts = Counter(tokens)
    
    # Calculate unigram probabilities
    unigram_probs = {word: count / total_tokens for word, count in word_counts.items()}
    
    return word_counts, unigram_probs

def unigram_word_prob(word, unigram_probs):
    return unigram_probs.get(word, 0.0)

def unigram_sentence_prob(sentence, unigram_probs):
    tokens = tokenize(sentence)
    prob = 1.0
    for token in tokens:
        p = unigram_word_prob(token, unigram_probs)
        if p == 0:
            return 0.0  # Unseen word
        prob *= p
    return prob

def build_bigram_model(corpus):
    tokens = tokenize(corpus)
    unigram_counts = Counter(tokens)
    bigram_counts = defaultdict(Counter)
    
    for i in range(len(tokens)-1):
        w1, w2 = tokens[i], tokens[i+1]
        bigram_counts[w1][w2] += 1
    
    # Calculate conditional probabilities P(w2 | w1)
    bigram_probs = {}
    for w1 in bigram_counts:
        total_count = sum(bigram_counts[w1].values())
        bigram_probs[w1] = {w2: count / total_count for w2, count in bigram_counts[w1].items()}
    
    return unigram_counts, bigram_probs

def bigram_word_prob(w1, w2, bigram_probs):
    return bigram_probs.get(w1, {}).get(w2, 0.0)

def bigram_sentence_prob(sentence, unigram_counts, bigram_probs):
    tokens = tokenize(sentence)
    if len(tokens) == 0:
        return 0.0
    
    # Starting probability: Assume P(w1) = count(w1)/total
    total_unigrams = sum(unigram_counts.values())
    prob = unigram_counts.get(tokens[0], 0) / total_unigrams
    
    for i in range(1, len(tokens)):
        w1, w2 = tokens[i-1], tokens[i]
        p = bigram_word_prob(w1, w2, bigram_probs)
        if p == 0:
            return 0.0  # unseen bigram
        prob *= p
    
    return prob

# Predict next word based on previous word
def predict_next_word(prev_word, bigram_probs):
    if prev_word not in bigram_probs:
        return None
    next_words = list(bigram_probs[prev_word].keys())
    probs = list(bigram_probs[prev_word].values())
    return random.choices(next_words, weights=probs, k=1)[0]

def generate_sentence(prompt, bigram_probs, max_words=10):
    tokens = tokenize(prompt)
    if not tokens:
        return ""
    
    current_word = tokens[-1]
    generated = tokens.copy()
    
    for _ in range(max_words):
        next_word = predict_next_word(current_word, bigram_probs)
        if not next_word:
            break
        generated.append(next_word)
        current_word = next_word
    
    return " ".join(generated)


