# UIpages/tokenizer.py
import streamlit as st
import re
from nlpFunc.nltk_tokenizers import (
    word_tokenizer,
    sentence_tokenizer,
    generate_ngrams,
    cap_tokenizer,
    general_regexp_tokenizer,
)


def show_tokenizer_ui():
    # ---------- Text input --------------------------------------------------
    st.title("🧠 NLTK Tokenizer Playground")

    text = st.text_area(
        "Enter your text below:",
        height=150,
        value="Should I pick up some black-eyed peas as well?",
    )

    # ---------- Session‑state toggles --------------------------------------
    for key in ("show_ngram_input", "show_regex_input"):
        if key not in st.session_state:
            st.session_state[key] = False

    def _reset_toggles():
        st.session_state.show_ngram_input = False
        st.session_state.show_regex_input = False

    def _safe_tokenize(tokenizer_fn, header):
        """Run *tokenizer_fn* on *text* if it is not empty, else warn."""
        if text.strip():
            tokens = tokenizer_fn(text)
            st.subheader(header)
            st.write(tokens)
        else:
            st.error("Please enter some text to tokenize.")

    # ---------- Action buttons ---------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("🔤 Word Tokenize"):
            _reset_toggles()
            _safe_tokenize(word_tokenizer, "Word Tokens")

    with col2:
        if st.button("📑 Sentence Tokenize"):
            _reset_toggles()
            _safe_tokenize(sentence_tokenizer, "Sentence Tokens")

    with col3:
        if st.button("🔗 N‑gram (Ngrams)"):
            _reset_toggles()
            st.session_state.show_ngram_input = True

    with col4:
        if st.button("🆙 Capitalized Tokenize"):
            _reset_toggles()
            _safe_tokenize(cap_tokenizer, "Capitalized Tokens")

    with col5:
        if st.button("🔧 RegExp Tokenize"):
            _reset_toggles()
            st.session_state.show_regex_input = True

    # ---------- N‑gram interface -------------------------------------------
    if st.session_state.show_ngram_input:
        n = st.number_input(
            "Enter a number for N (for N‑grams)", min_value=1, max_value=10, step=1
        )
        if text.strip():
            n_grams = generate_ngrams(text, n)
            st.subheader(f"{n}-Grams")
            st.write(n_grams)
        else:
            st.error("Please enter some text to tokenize.")

    # ---------- RegExp interface -------------------------------------------
    if st.session_state.show_regex_input:
        pattern = st.text_input("Enter a regular expression pattern", value="\\w+")
        if pattern:  # Avoid running on empty pattern
            if text.strip():
                try:
                    tokens = general_regexp_tokenizer(text, pattern)
                    st.subheader("RegExp Tokens")
                    st.write(tokens)
                except re.error as err:
                    st.error(f"❌ Regex error: {err}")
                except RuntimeError as err:
                    st.error(f"⚠️ {err}")
            else:
                st.error("Please enter some text to tokenize.")

