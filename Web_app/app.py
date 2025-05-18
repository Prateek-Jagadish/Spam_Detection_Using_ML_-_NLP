import streamlit as st
import pickle
import re
import fitz  # PyMuPDF
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

def extract_entities(text):
    doc = nlp(text)
    extracted = {}
    for ent in doc.ents:
        extracted.setdefault(ent.label_, []).append(ent.text)
    return extracted if extracted else None

def read_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception:
        return None

# --- UI ---
st.set_page_config(page_title="Spade - Smart Spam Detector", layout="centered")
st.title("🛡️ Spade - AI-powered Spam Detection")

st.markdown("**Welcome to Spade!** Upload a message or paste text to detect **spam** and discover key **entities**.")

upload_tab, paste_tab = st.tabs(["📄 Upload File (.pdf/.txt)", "✍️ Paste Text"])

text_input = None

with upload_tab:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file:", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text_input = read_pdf(uploaded_file)
        else:
            text_input = uploaded_file.read().decode("utf-8")

        if not text_input:
            st.error("❌ Could not extract text from the file.")
        else:
            st.success("✅ File uploaded successfully!")

with paste_tab:
    text_input_manual = st.text_area("Type or paste your message here:", height=250)
    if text_input_manual.strip():
        text_input = text_input_manual

if text_input:
    if st.button("🚀 Run Detection"):
        with st.spinner("🔍 Analyzing message..."):
            cleaned = clean_text(text_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][1]

            label = "Spam" if pred == 1 else "Ham"
            confidence = f"{prob * 100:.2f}%"

        st.success(f"### 🔎 Prediction: **{label}**")
        st.markdown(f"**🎯 Confidence:** {confidence}")

        tabs = st.tabs(["📊 Spam Probability", "🧠 Named Entities"])
        with tabs[0]:
            st.subheader("Probability Distribution")
            fig, ax = plt.subplots()
            sns.barplot(x=["Ham", "Spam"], y=model.predict_proba(vec)[0], ax=ax, palette="coolwarm")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        with tabs[1]:
            st.subheader("Entities Found")
            entity_dict = extract_entities(text_input)
            if entity_dict:
                for label, values in entity_dict.items():
                    with st.expander(f"{label} ({spacy.explain(label)})"):
                        st.write(", ".join(set(values)))
            else:
                st.info("No named entities found.")
else:
    st.info("👆 Upload a file or paste your message above.")
