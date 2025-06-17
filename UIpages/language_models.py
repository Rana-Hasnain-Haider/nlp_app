import streamlit as st
from nlpFunc.language_models import (
    build_unigram_model,
    unigram_word_prob,
    unigram_sentence_prob,
    build_bigram_model,
    bigram_word_prob,
    bigram_sentence_prob,
    predict_next_word,
    generate_sentence,
)

def show_ngram_ui():
    st.title("🧮 Unigram & Bigram Language Models")

    st.subheader("Step 1: Input Text (Max 1500 Words)")
    user_text = st.text_area("Paste your corpus here:", height=300)

    def limit_words(text, limit=1500):
        tokens = text.split()
        return " ".join(tokens[:limit])

    if user_text:
        word_count = len(user_text.split())
        if word_count > 1500:
            st.warning(f"⚠️ You entered {word_count} words. Only the first 1500 words will be used.")
            user_text = limit_words(user_text)

    if "unigram_data" not in st.session_state:
        st.session_state.unigram_data = None
    if "bigram_data" not in st.session_state:
        st.session_state.bigram_data = None

    st.subheader("Step 2: Choose Model Type")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📗 Build Unigram Model"):
            word_counts, unigram_probs = build_unigram_model(user_text)
            st.session_state.unigram_data = {"counts": word_counts, "probs": unigram_probs}
            st.session_state.bigram_data = None  # 🧹 Clear the other model
            st.success("✅ Unigram model built successfully!")

    with col2:
        if st.button("📘 Build Bigram Model"):
            unigram_counts, bigram_probs = build_bigram_model(user_text)
            st.session_state.bigram_data = {"counts": unigram_counts, "probs": bigram_probs}
            st.session_state.unigram_data = None  # 🧹 Clear the other model
            st.success("✅ Bigram model built successfully!")


    st.markdown("---")

    if st.session_state.unigram_data:
        st.subheader("🔍 Unigram Model Operations")

        word = st.text_input("🔤 Enter a word to get its probability (Unigram):")
        if word:
            prob = unigram_word_prob(word.lower(), st.session_state.unigram_data["probs"])
            st.write(f"**Probability of '{word}':** {prob:.6f}")

        sentence = st.text_input("📝 Enter a sentence to get its probability (Unigram):")
        if sentence:
            prob = unigram_sentence_prob(sentence, st.session_state.unigram_data["probs"])
            st.write(f"**Probability of sentence:** {prob:.10f}")

    st.markdown("---")

    if st.session_state.bigram_data:
        st.subheader("🔎 Bigram Model Operations")

        word_pair = st.text_input("🔤 Enter two words (space-separated) to get their bigram probability:")
        if word_pair:
            parts = word_pair.lower().split()
            if len(parts) == 2:
                w1, w2 = parts
                prob = bigram_word_prob(w1, w2, st.session_state.bigram_data["probs"])
                st.write(f"**P({w2} | {w1}) =** {prob:.6f}")
            else:
                st.warning("Please enter exactly two words.")

        bigram_sentence = st.text_input("📝 Enter a sentence to get its probability (Bigram):")
        if bigram_sentence:
            prob = bigram_sentence_prob(
                bigram_sentence,
                st.session_state.bigram_data["counts"],
                st.session_state.bigram_data["probs"]
            )
            st.write(f"**Probability of sentence:** {prob:.10f}")

        st.markdown("### 🔮 Bigram Predictions")

        next_word_prompt = st.text_input("✨ Enter a word to predict the next word:")
        if next_word_prompt:
            next_word = predict_next_word(next_word_prompt.lower(), st.session_state.bigram_data["probs"])
            if next_word:
                st.write(f"**Predicted next word after '{next_word_prompt}':** {next_word}")
            else:
                st.write("❌ No prediction available for that word.")

        sentence_gen_prompt = st.text_input("💡 Enter a sentence prompt to generate a continuation:")
        if sentence_gen_prompt:
            gen_sentence = generate_sentence(
                sentence_gen_prompt,
                st.session_state.bigram_data["probs"],
                max_words=10
            )
            st.write(f"**Generated Sentence:** {gen_sentence}")
