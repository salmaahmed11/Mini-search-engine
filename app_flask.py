import re
import pickle
import os
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import google.generativeai as genai

from flask import Flask, render_template, request, jsonify

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIG (relative paths so it works on any server) ─────
CSV_PATH        = "medicine_cleaned133(in).csv"
TEMPLATE_FOLDER = "html_files"
CACHE_FILE      = "cache.pkl"
USE_CSV         = True
app    = Flask(__name__, template_folder=TEMPLATE_FOLDER)

# Key left exactly as provided
GOOGLE_API_KEY = "AIzaSyCBic9jEwXsuvj1m5Qyw5mmRdTt4eIp-MIx"
genai.configure(api_key=GOOGLE_API_KEY)
client = genai.GenerativeModel("models/gemini-2.5-flash")

# -----------------------------
# HELPER: find a column case-insensitively
# -----------------------------
def find_col(df, *candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    raise KeyError(
        f"None of {candidates} found in CSV columns: {df.columns.tolist()}"
    )


# -----------------------------
# DATA LOADING FROM CSV
# -----------------------------
def load_csv_docs(csv_path):
    df = pd.read_csv(csv_path, encoding='latin1')
    print("Columns found:", df.columns.tolist())

    col_name  = find_col(df, "name", "medicine_name", "drug_name", "medicine", "drug")
    col_class = find_col(df, "therapeutic_class", "class", "category", "drug_class")
    col_use   = find_col(df, "use", "uses", "indication", "indications")
    col_side  = find_col(df, "side_effects", "side effect", "sideeffects", "adverse_effects")

    print(f"Using columns -> name='{col_name}' | class='{col_class}' | use='{col_use}' | side_effects='{col_side}'")

    df["document"] = (
        df[col_name].fillna("")  + " " +
        df[col_class].fillna("") + " " +
        df[col_use].fillna("")   + " " +
        df[col_side].fillna("")
    )

    docs = {i: row["document"] for i, row in df.iterrows()}
    return docs, df


# -----------------------------
# WEB CRAWLER (fallback)
# -----------------------------
def crawl_web(seed_urls, max_pages=5):
    docs    = {}
    doc_id  = 1
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in seed_urls:
        if doc_id > max_pages:
            break
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            text = re.sub(r'\s+', ' ', soup.get_text(separator=" ")).strip()
            if len(text) > 300:
                docs[doc_id] = text[:3000]
                doc_id += 1
        except Exception:
            continue

    return docs


# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens     = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens     = [w for w in tokens if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens     = [lemmatizer.lemmatize(w) for w in tokens]
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
# TF-IDF (optimized)
# -----------------------------
def build_tfidf(docs):
    if not docs:
        docs = {1: "default text for search engine testing system"}

    doc_ids = list(docs.keys())
    texts   = list(docs.values())

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        max_features=50000,
        min_df=2,
        max_df=0.95
    )
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix, doc_ids, texts


# -----------------------------
# SEARCH ENGINE  (fast O(1) lookup)
# -----------------------------
def search(query, docs, index, vectorizer, matrix, doc_ids, texts):
    tokens     = preprocess_text(query)
    candidates = set()
    for t in tokens:
        candidates |= index.get(t, set())

    if not candidates:
        return []

    indices = [doc_id_to_index[d] for d in candidates if d in doc_id_to_index]
    if not indices:
        return []

    query_vec = vectorizer.transform([query])
    sims      = cosine_similarity(query_vec, matrix[indices]).flatten()

    results = sorted(
        zip([doc_ids[i] for i in indices],
            [texts[i]   for i in indices],
            sims),
        key=lambda x: x[2],
        reverse=True
    )[:20]
    return results


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_results(results, relevant_flags, k=5):
    if not results:
        return 0, 0

    k = min(k, len(results))
    relevant_docs = [
        doc_id
        for (doc_id, _, _), flag in zip(results[:k], relevant_flags)
        if flag == "relevant"
    ]

    precision      = len(relevant_docs) / k
    total_relevant = relevant_flags.count("relevant")
    recall         = len(relevant_docs) / total_relevant if total_relevant > 0 else 0

    return precision, recall


# -----------------------------
# gemini CHATBOT with CSV context
# -----------------------------
def chat_with_gemini(user_message, conversation_history):
    csv_results = search(
        user_message, documents, inverted_index,
        vectorizer, matrix, doc_ids, texts
    )

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

    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in conversation_history]
    )

    response = client.generate_content([
        {"text": system_prompt},
        {"text": history_text},
        {"text": f"User: {user_message}"}
    ])

    return response.text

# ─── LOAD OR BUILD CACHE AT STARTUP ───────────────────────
def load_or_build():
    if os.path.exists(CACHE_FILE):
        print("Loading from cache... (fast)")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Building index for the first time (this may take a few minutes)...")

    if USE_CSV:
        documents, medicine_df = load_csv_docs(CSV_PATH)
        print(f"Loaded {len(documents)} medicines from CSV.")
    else:
        seeds = [
            "https://en.wikipedia.org/wiki/Isoniazid/rifampicin",
            "https://en.wikipedia.org/wiki/Insulin_(medication)",
            "https://en.wikipedia.org/wiki/Aspirin",
            "https://en.wikipedia.org/wiki/Paracetamol",
        ]
        documents   = crawl_web(seeds, max_pages=10)
        medicine_df = None
        print(f"Crawled {len(documents)} docs from the web.")

    inverted_index                     = build_inverted_index(documents)
    vectorizer, matrix, doc_ids, texts = build_tfidf(documents)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((documents, medicine_df, inverted_index,
                     vectorizer, matrix, doc_ids, texts), f)
    print("Cache saved! Next startup will be instant.")

    return documents, medicine_df, inverted_index, vectorizer, matrix, doc_ids, texts


# ─── STARTUP ──────────────────────────────────────────────
documents, medicine_df, inverted_index, vectorizer, matrix, doc_ids, texts = load_or_build()
doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}


# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    results   = []
    precision = None
    recall    = None

    if request.method == "POST":
        if "query" in request.form:
            query   = request.form["query"]
            results = search(query, documents, inverted_index,
                             vectorizer, matrix, doc_ids, texts)
            results = results[:20]

        elif "evaluate" in request.form:
            doc_ids_list = request.form.getlist("doc_ids")
            scores_list  = request.form.getlist("scores")
            texts_list   = request.form.getlist("texts")

            results = list(zip(
                list(map(int,   doc_ids_list)),
                texts_list,
                list(map(float, scores_list))
            ))

            relevance = [
                "relevant" if request.form.get(f"relevant_{doc_id}") == "1"
                else "not_relevant"
                for doc_id in doc_ids_list
            ]

            precision, recall = evaluate_results(results, relevance)

    return render_template("index.html",
                           results=results,
                           precision=precision,
                           recall=recall)


@app.route("/chat", methods=["POST"])
def chat():
    data                 = request.get_json()
    user_message         = data.get("message", "").strip()
    conversation_history = data.get("history", [])

    if not user_message:
        return jsonify({"reply": "Please type a message."})

    try:
        reply = chat_with_gemini(user_message, conversation_history)
    except Exception as e:
        reply = f"Error contacting Gemini: {str(e)}"

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
