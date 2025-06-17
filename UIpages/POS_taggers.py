import streamlit as st
from nlpFunc.POS_taggers import (
    rule_based,
    HMM_tagger,
    Maximum_Entropy_tagger,
    brill_tagger,
    Logistic_Regression_tagger,
)

# Optional: Define colors for different POS tags
TAG_COLORS = {
    "NN": "#FFEEAD", "NNS": "#FFEEAD", "NNP": "#FFEEAD", "NNPS": "#FFEEAD",
    "VB": "#AEDFF7", "VBD": "#AEDFF7", "VBG": "#AEDFF7", "VBN": "#AEDFF7", "VBP": "#AEDFF7", "VBZ": "#AEDFF7",
    "JJ": "#B2F7B8", "JJR": "#B2F7B8", "JJS": "#B2F7B8",
    "RB": "#F7C3B2", "RBR": "#F7C3B2", "RBS": "#F7C3B2",
    "DT": "#E0D3F7", "IN": "#D3F7F3", "CD": "#FCE3A1",
    "POS": "#FFC1CC", "MD": "#C1E1C1", "PRP": "#F7D6E0", "PRP$": "#F7D6E0"
}

def color_word(word, tag):
    color = TAG_COLORS.get(tag, "#F0F0F0")  # Default gray if tag not found
    return f"<span style='background-color:{color}; padding:4px; margin:2px; border-radius:5px; display:inline-block;'>{word} <sub><code>{tag}</code></sub></span>"

def show_pos_tagger_ui():
    st.title("🏷️ POS Taggers")
    st.markdown("Enter a sentence and choose a POS tagging model to tag each word.")

    user_input = st.text_area("✏️ Enter sentence here:", height=100, max_chars=500)

    st.subheader("🔧 Choose a Tagging Model")
    selected_model = st.selectbox("Select a POS Tagger:", [
        "Rule-Based Tagger",
        "HMM Tagger",
        "Maximum Entropy Tagger",
        "Brill Tagger",
        "Logistic Regression Tagger"
    ])

    if st.button("🚀 Tag Sentence"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a sentence.")
        else:
            try:
                if selected_model == "Rule-Based Tagger":
                    result = rule_based(user_input)

                elif selected_model == "HMM Tagger":
                    result = HMM_tagger(user_input)

                elif selected_model == "Maximum Entropy Tagger":
                    result = Maximum_Entropy_tagger(user_input)

                elif selected_model == "Brill Tagger":
                    result = brill_tagger(user_input)

                elif selected_model == "Logistic Regression Tagger":
                    result = Logistic_Regression_tagger(user_input)

                # Display tagged output
                st.success(f"✅ Tagged with {selected_model}")
                st.markdown("### 🖍️ Colored Tagged Sentence:")
                colored = " ".join([color_word(word, tag) for word, tag in result])
                st.markdown(colored, unsafe_allow_html=True)

                st.markdown("### 📋 Detailed Tags Table:")
                st.table({"Word": [w for w, _ in result], "POS Tag": [t for _, t in result]})

            except Exception as e:
                st.error(f"❌ Error: {e}")
