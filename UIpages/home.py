import streamlit as st

def show_tokenizer_home():
    st.header("📚 Tokenizer Function Overview")
    st.write("Welcome! This section gives a brief NLP-centric explanation of the available functionalities available in this app.")

    st.markdown("---")

    st.subheader("🔹 Word Tokenizer")
    st.markdown("""
    **Purpose**:  
    Splits text into individual words using rules for whitespace and punctuation.  
    It forms the foundational step for most NLP pipelines by isolating lexical units (tokens) from raw text.  
    Enables tasks like word frequency analysis, POS tagging, and embeddings.
    """)

    st.subheader("🔹 Sentence Tokenizer")
    st.markdown("""
    **Purpose**:  
    Breaks down long text into sentences using punctuation and language-specific heuristics.  
    Useful for document segmentation, summarization, and context-based processing.  
    Maintains sentence boundaries which help in syntactic and semantic analysis.
    """)

    st.subheader("🔹 N-Gram Generator")
    st.markdown("""
    **Purpose**:  
    Creates sequences of *n* contiguous words (n-grams) from tokenized text.  
    Crucial for capturing context, co-occurrence, and building statistical models.  
    Often used in language modeling, autocomplete systems, and spelling correction.
    """)

    st.subheader("🔹 Capitalized Word Tokenizer")
    st.markdown("""
    **Purpose**:  
    Extracts words that start with a capital letter using regular expressions.  
    Helps identify proper nouns or named entities like people, places, or brands.  
    Supports Named Entity Recognition (NER) and headline parsing.
    """)

    st.subheader("🔹 Custom RegEx Tokenizer")
    st.markdown("""
    **Purpose**:  
    Allows users to define their own rules for tokenization via regular expressions.  
    Facilitates domain-specific or language-specific preprocessing where standard tokenizers fail.  
    Offers granular control over what is considered a "token" in a given context.
    """)

    st.markdown("---")

import streamlit as st

def show_preprocessing_home():
    st.header("🧹 Preprocessing-I Function Overview")
    st.write("This Section summarizes the key preprocessing functions used in NLP tasks. Each function prepares raw text for better analysis or modeling.")

    st.markdown("---")

    st.subheader("🔸 Remove Punctuation (with Whitespace)")
    st.markdown("""
    **Purpose**:  
    Replaces all punctuation marks with spaces to preserve token boundaries.  
    Helps in maintaining word separation while cleaning noisy text.  
    Ensures punctuation does not affect token count or vocabulary size.
    """)

    st.subheader("🔸 Remove Digits")
    st.markdown("""
    **Purpose**:  
    Eliminates words that contain numeric digits to reduce noise.  
    Useful when numbers are irrelevant to the analysis or model (e.g., names, reviews).  
    Prevents digits from skewing vector space or word frequencies.
    """)

    st.subheader("🔸 Remove Stopwords")
    st.markdown("""
    **Purpose**:  
    Filters out commonly used words (like "the", "is", "and") that add little semantic value.  
    Helps models focus on more informative words.  
    Improves performance of frequency-based methods like Bag of Words or TF-IDF.
    """)

    st.subheader("🔸 Synonym Expansion")
    st.markdown("""
    **Purpose**:  
    Enhances a user query by adding synonyms for meaningful words using WordNet.  
    Expands search or matching coverage for information retrieval and semantic search.  
    Avoids over-expansion by skipping stopwords and duplicates.
    """)

def show_preprocessing2_home():
    st.header("🧹 Preprocessing-II Function Overview")
    st.write("This Section provides NLP-focused explanations of various essential preprocessing techniques used for text normalization and extraction.")

    st.markdown("---")

    st.subheader("🌿 Lancaster Stemmer")
    st.markdown("""
    **Purpose**:  
    Reduces words to their base forms using a rule-based, aggressive stemming algorithm.  
    Particularly useful when you want strong normalization for tasks like document classification.  
    May over-stem and remove distinctions between words.
    """)

    st.subheader("🌱 Porter Stemmer")
    st.markdown("""
    **Purpose**:  
    A widely-used algorithm that reduces words to their morphological root.  
    It’s less aggressive than Lancaster and often preferred for applications like search engines.  
    Helps in collapsing word variants like “running”, “runs”, and “run” into a single token.
    """)

    st.subheader("🔤 WordNet Lemmatizer")
    st.markdown("""
    **Purpose**:  
    Converts words to their dictionary base form (lemma) based on POS (e.g., “was” → “be”).  
    Preserves linguistic meaning better than stemming.  
    Useful in knowledge representation, reasoning, and sophisticated NLP pipelines.
    """)

    st.subheader("🧑‍💼 Named Entity Recognition (NER)")
    st.markdown("""
    **Purpose**:  
    Detects and classifies named entities such as persons, organizations, locations, and dates.  
    Helps in understanding the real-world references within a sentence.  
    Critical for information extraction, knowledge graph building, and summarization.
    """)


def show_segmenter_description():
    st.header("✂️ Semantic Text Segmentation")
    
    st.markdown("""
    **What is This Page About?**

    The Transformer-based segmenter identifies **topic shifts** in a long paragraph by analyzing the **semantic meaning** of sentences using **pretrained transformer models**.

    ---
    ### 🔍 How It Works:

    1. **Sentence Embedding:**  
       Each sentence is embedded using the SentenceTransformer model (`all-MiniLM-L6-v2`), capturing its semantic context.

    2. **Rolling Similarity Check:**  
       For each sentence, we compare it with the **average of the previous 3 sentences** using **cosine similarity**.

    3. **Detecting Topic Shifts:**  
       If the similarity score drops significantly (below a dynamic threshold), it's treated as a **boundary**, marking a shift in topic or meaning.

    4. **Smoothing for Robustness:**  
       To avoid false positives, we smooth the similarity scores using a moving average and enforce a minimum gap between segments.

    5. **Text Segmentation:**  
       The final segments contain grouped sentences that are semantically similar, helping you understand text structure and sub-topics clearly.

    """)

def show_language_model_description():
    st.header("📚 Unigram & Bigram Language Models")
    
    # -------------------------
    st.header("🔹 Unigram Model")
    st.markdown("""
    The **Unigram Model** assumes that each word in a sentence is independent of the previous words. It helps estimate the **probability of individual words** and compute the **likelihood of entire sentences** based on word frequency.

    - **`tokenize(text)`**  
      Splits the input corpus into lowercase tokens.

    - **`build_unigram_model(corpus)`**  
      Builds a frequency-based model assigning probabilities to each word in the corpus.

    - **`unigram_word_prob(word, probs)`**  
      Looks up the probability of a specific word.

    - **`unigram_sentence_prob(sentence, probs)`**  
      Multiplies the probabilities of individual words to get the overall sentence likelihood.
    """)

    # -------------------------
    st.header("🔹 Bigram Model")
    st.markdown("""
    The **Bigram Model** captures word **dependencies** by considering the **probability of a word given the previous word**. This gives better contextual understanding than the unigram model.

    - **`build_bigram_model(corpus)`**  
      Constructs bigram counts and conditional probabilities like P(w₂ | w₁) based on word pairs.

    - **`bigram_word_prob(w1, w2, probs)`**  
      Retrieves the conditional probability of a word following another.

    - **`bigram_sentence_prob(sentence, unigram_counts, bigram_probs)`**  
      Combines unigram and bigram probabilities to compute the sentence's overall likelihood.

    - **`predict_next_word(prev_word, bigram_probs)`**  
      Uses the bigram model to predict the most likely next word after a given word.

    - **`generate_sentence(prompt, bigram_probs)`**  
      Starts with a prompt and auto-generates a sentence by sampling probable next words.
    """)


def show_pos_tagger_home():
    st.header("🏷️ POS Taggers Overview")
    st.write("This section provides a brief summary of different Part-of-Speech (POS) tagging techniques used in Natural Language Processing (NLP). Each model tags words in a sentence with their grammatical roles.")

    st.markdown("---")

    st.subheader("🔸 Rule-Based Tagger")
    st.markdown("""
    **Purpose**:  
    Applies handcrafted linguistic rules to identify the POS tag of each word.  
    **Pros**: Fast, interpretable, no training needed.  
    **Cons**: Doesn't generalize well; struggles with ambiguous or unseen words.
    """)

    st.subheader("🔸 Hidden Markov Model (HMM) Tagger")
    st.markdown("""
    **Purpose**:  
    Uses probabilistic models to assign the most likely tag sequence based on word transitions.  
    **Pros**: Captures sequential context, better accuracy than rule-based.  
    **Cons**: Limited feature support, assumptions may reduce performance.
    """)

    st.subheader("🔸 Maximum Entropy Tagger")
    st.markdown("""
    **Purpose**:  
    Uses a statistical model that incorporates a wide variety of contextual features.  
    **Pros**: Flexible and feature-rich, handles complex tagging better.  
    **Cons**: Slower and requires labeled data for training.
    """)

    st.subheader("🔸 Brill Tagger")
    st.markdown("""
    **Purpose**:  
    Combines rule-based and statistical learning by learning transformation rules from tagged data.  
    **Pros**: Interpretable and hybrid in nature.  
    **Cons**: Training takes longer and may overfit small datasets.
    """)

    st.subheader("🔸 Logistic Regression Tagger")
    st.markdown("""
    **Purpose**:  
    Uses supervised machine learning to predict tags based on features like word identity, shape, and neighbors.  
    **Pros**: Simple, accurate, works well with feature engineering.  
    **Cons**: Ignores word order; lacks sequential modeling unless extended (e.g., CRF).
    """)
