import re
import pickle
import os
from collections import defaultdict

import streamlit as st
import pandas as pd
import google.generativeai as genai

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── NLTK SETUP (safe for cloud) ───────────────────────────
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# ─── CONFIG (relative paths — files must sit in repo root) ─
CSV_PATH = "medicine_cleaned133(in).csv"
CACHE_FILE = "cache.pkl"
USE_CSV = True

# ─── API KEY — never hardcode. Use Streamlit secrets. ──────
# In Streamlit Cloud: Settings -> Secrets -> add:
# GOOGLE_API_KEY = "your_new_key_here"
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

if not GOOGLE_API_KEY:
    st.warning("No Gemini API key found. Add GOOGLE_API_KEY to your Streamlit secrets to enable the chatbot.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

client = genai.GenerativeModel("models/gemini-2.5-flash") if GOOGLE_API_KEY else None


# -----------------------------
# HELPER: find a column case-insensitively
# -----------------------------
def find_col(df, *candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    raise KeyError(f"None of {candidates} found in CSV columns: {df.columns.tolist()}")


# -----------------------------
# DATA LOADING FROM CSV
# -----------------------------
def load_csv_docs(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")

    col_name = find_col(df, "name", "medicine_name", "drug_name", "medicine", "drug")
    col_class = find_col(df, "therapeutic_class", "class", "category", "drug_class")
    col_use = find_col(df, "use", "uses", "indication", "indications")
    col_side = find_col(df, "side_effects", "side effect", "sideeffects", "adverse_effects")

    df["document"] = (
        df[col_name].fillna("") + " "
        + df[col_class].fillna("") + " "
        + df[col_use].fillna("") + " "
        + df[col_side].fillna("")
    )

    docs = {i: row["document"] for i, row in df.iterrows()}
    return docs, df


# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens


def tokenizer(text):
    return preprocess_text(text)


# -----------------------------
# INVERTED INDEX
# -----------------------------
def build_inverted_index(docs):
    index = defaultdict(set)
    for doc_id, text in docs.items():
        for t in preprocess_text(text):
            index[t].add(doc_id)
    return index


# -----------------------------
# TF-IDF
# -----------------------------
def build_tfidf(docs):
    if not docs:
        docs = {1: "default text for search engine testing system"}

    doc_ids = list(docs.keys())
    texts = list(docs.values())

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer, max_features=50000, min_df=2, max_df=0.95
    )
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix, doc_ids, texts


# -----------------------------
# SEARCH
# -----------------------------
def search(query, docs, index, vectorizer, matrix, doc_ids, texts, doc_id_to_index):
    tokens = preprocess_text(query)
    candidates = set()
    for t in tokens:
        candidates |= index.get(t, set())

    if not candidates:
        return []

    indices = [doc_id_to_index[d] for d in candidates if d in doc_id_to_index]
    if not indices:
        return []

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix[indices]).flatten()

    results = sorted(
        zip([doc_ids[i] for i in indices], [texts[i] for i in indices], sims),
        key=lambda x: x[2],
        reverse=True,
    )[:20]
    return results


# -----------------------------
# GEMINI CHAT
# -----------------------------
def chat_with_gemini(user_message, conversation_history, documents, inverted_index,
                      vectorizer, matrix, doc_ids, texts, doc_id_to_index):
    csv_results = search(user_message, documents, inverted_index, vectorizer,
                          matrix, doc_ids, texts, doc_id_to_index)

    if csv_results:
        context = "Relevant medicines from our database:\n"
        for i, (doc_id, text, score) in enumerate(csv_results[:5], 1):
            context += f"{i}. {text[:300]}\n"
    else:
        context = "No specific medicines found in the database for this query."

    system_prompt = (
        "You are a helpful medical assistant specialized in medicines. "
        "You have access to a medicine database. "
        "When answering, use the database context provided if relevant, "
        "and also use your general medical knowledge. "
        "Always be clear, accurate, and remind users to consult a doctor for medical decisions.\n\n"
        + context
    )

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])

    response = client.generate_content([
        {"text": system_prompt},
        {"text": history_text},
        {"text": f"User: {user_message}"},
    ])

    return response.text


# ─── LOAD OR BUILD INDEX (cached across reruns) ────────────
@st.cache_resource
def load_or_build():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    documents, medicine_df = load_csv_docs(CSV_PATH)
    inverted_index = build_inverted_index(documents)
    vectorizer, matrix, doc_ids, texts = build_tfidf(documents)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((documents, medicine_df, inverted_index, vectorizer, matrix, doc_ids, texts), f)

    return documents, medicine_df, inverted_index, vectorizer, matrix, doc_ids, texts


# ─── STREAMLIT UI ───────────────────────────────────────────
st.set_page_config(page_title="Medicine Search Engine", page_icon="💊", layout="wide")
st.title("💊 Medicine Search Engine")
st.caption("Search 224,000+ medicines by name, condition, therapeutic class, or side effects — with an AI assistant powered by Gemini.")

documents, medicine_df, inverted_index, vectorizer, matrix, doc_ids, texts = load_or_build()
doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

tab_search, tab_chat = st.tabs(["🔍 Search", "💬 Ask the AI Assistant"])

with tab_search:
    query = st.text_input("Search for a medicine, condition, or side effect")
    if query:
        results = search(query, documents, inverted_index, vectorizer, matrix,
                          doc_ids, texts, doc_id_to_index)
        if not results:
            st.info("No results found.")
        else:
            for doc_id, text, score in results:
                with st.container(border=True):
                    st.write(text)
                    st.caption(f"Relevance score: {score:.3f}")

with tab_chat:
    if not client:
        st.info("Add a Gemini API key to Streamlit secrets to enable the chatbot.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_msg = st.chat_input("Ask about a medicine...")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = chat_with_gemini(
                            user_msg, st.session_state.chat_history, documents,
                            inverted_index, vectorizer, matrix, doc_ids, texts, doc_id_to_index
                        )
                    except Exception as e:
                        reply = f"Error contacting Gemini: {e}"
                    st.write(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
