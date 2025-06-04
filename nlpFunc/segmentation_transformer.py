import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def compute_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    """Computes sentence embeddings using a transformer model."""
    model = SentenceTransformer(model_name)
    return model.encode(sentences)


def compute_similarities(embeddings, window_size=3):
    """Computes similarity between rolling average and current sentence embedding."""
    similarities = []
    for i in range(window_size, len(embeddings)):
        prev_mean = np.mean(embeddings[i - window_size:i], axis=0)
        similarity = cosine_similarity([prev_mean], [embeddings[i]])[0][0]
        similarities.append(similarity)
    return similarities


def find_segments(similarities, window_size, min_gap=2, smoothing_window=3, offset=0.05):
    """Finds segment boundaries based on smoothed similarity drops."""
    smoothed = np.convolve(similarities, np.ones(smoothing_window)/smoothing_window, mode='same')
    threshold = np.mean(smoothed) - offset

    segment_indices = [0]
    for i, sim in enumerate(smoothed):
        if sim < threshold:
            idx = i + window_size
            if idx - segment_indices[-1] >= min_gap:
                segment_indices.append(idx)
    return segment_indices


def segment_text(sentences, segment_indices):
    """Groups sentences into segments based on boundary indices."""
    segment_indices.append(len(sentences))
    return [
        " ".join(sentences[start:end])
        for start, end in zip(segment_indices[:-1], segment_indices[1:])
    ]




def segment(text):
    sentences = nltk.sent_tokenize(text)
    embeddings = compute_embeddings(sentences)
    similarities = compute_similarities(embeddings, window_size=3)
    segment_indices = find_segments(similarities, window_size=3)
    segments = segment_text(sentences, segment_indices)
    return segments
   
