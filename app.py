import streamlit as st
from UIpages.tokenizer import show_tokenizer_ui
from UIpages.preprocessings import show_preprocessing_ui
from UIpages.preprocessings2 import show_preprocessing_ui2
from UIpages.transformer_segmenter import show_segmenter_ui
from UIpages.language_models import show_ngram_ui
from UIpages.POS_taggers import show_pos_tagger_ui
from nltk.probability import LidstoneProbDist
from UIpages.home import(
    show_language_model_description,
    show_pos_tagger_home,
    show_preprocessing2_home,
    show_preprocessing_home,
    show_segmenter_description,
    show_tokenizer_home
)
def lidstone_estimator(fd, bins):
    return LidstoneProbDist(fd, 0.1, bins)


st.set_page_config(page_title="NLTK Playground", page_icon="🧠", layout="centered")

# Sidebar Navigation
st.sidebar.title("Navigation")

if "page" not in st.session_state:
    st.session_state.page = "home"

# Navigation buttons
if st.sidebar.button("Home"):
    st.session_state.page = "home"

if st.sidebar.button("🧪 Tokenizer"):
    st.session_state.page = "tokenizer"

if st.sidebar.button("🧹 Preprocessing-I"):
    st.session_state.page = "preprocessing-I"

if st.sidebar.button("🧹 Preprocessing-II"):
    st.session_state.page = "preprocessing-II"

if st.sidebar.button("✂️ TransformerSegmenter"):
    st.session_state.page = "segmenter"

if st.sidebar.button("🧮 N-Gram Language Models"):
    st.session_state.page = "ngram-models"

if st.sidebar.button("🏷️ POS Taggers"):
    st.session_state.page = "pos_taggers"



# Routing
if st.session_state.page == "tokenizer":
    show_tokenizer_ui()

elif st.session_state.page == "preprocessing-I":
    show_preprocessing_ui()

elif st.session_state.page == "preprocessing-II":
    show_preprocessing_ui2()

elif st.session_state.page == "segmenter":
    show_segmenter_ui()

elif st.session_state.page == "ngram-models":  # ← NEW route
    show_ngram_ui()

elif st.session_state.page == "pos_taggers":
    show_pos_tagger_ui()

else:
    st.title("👋 Welcome")
    st.write("Please select a page from the sidebar.")
    show_tokenizer_home()
    show_preprocessing_home()
    show_preprocessing2_home()
    show_segmenter_description()
    show_language_model_description()
    show_pos_tagger_home()
