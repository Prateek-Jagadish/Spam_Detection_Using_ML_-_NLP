import streamlit as st
import pickle
import re
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load ML model and vectorizer
with open("spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def ents(text):
    doc = nlp(text)
    extracted = {}
    for ent in doc.ents:
        if ent.label_ in extracted:
            extracted[ent.label_].append(ent.text)
        else:
            extracted[ent.label_] = [ent.text]
    return extracted if extracted else "no"

# Streamlit UI
st.set_page_config(page_title="Spade", layout="centered")
st.title("🛡️ Spade - Spam Detector")

st.write("Welcome to **Spade** - A Spam Detection system using Machine Learning and NLP.")

# Input options
text_input = st.text_area("✉️ Paste email or SMS text below:", height=300, placeholder="Type your message...")

if st.button("🚀 Detect"):
    if not text_input or len(text_input.strip()) < 10:
        st.error("Please enter at least 10 characters.")
    else:
        with st.spinner("Analyzing your message..."):
            cleaned = clean_text(text_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][1]  # spam probability

            label = "Spam" if pred == 1 else "Ham"
            confidence = f"{prob * 100:.2f}%"

            st.markdown(f"### 🔍 Prediction: **{label}**")
            st.markdown(f"### 🎯 Spam Confidence: **{confidence}**")

            # Bar plot
            st.subheader("Spam Probability")
            fig, ax = plt.subplots()
            sns.barplot(x=["Ham", "Spam"], y=model.predict_proba(vec)[0], ax=ax, palette="viridis")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            # Named Entity Extraction
            st.subheader("🧠 Named Entity Recognition")
            entity_dict = ents(text_input)
            if entity_dict == "no":
                st.write("No entities found.")
            else:
                for label, values in entity_dict.items():
                    with st.expander(f"{label} ({spacy.explain(label)})"):
                        st.write(", ".join(set(values)))





#To Run This: streamlit run app.py --logger.level=debug
