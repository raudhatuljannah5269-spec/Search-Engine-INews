import os
import string
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='../templates')

# --- 1. INISIALISASI SASTRAWI ---
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

# --- 2. LOAD DATA DARI COLAB (.pkl & .xlsx) ---
# Menggunakan os.path agar path file fleksibel saat dideploy ke Vercel
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load artikel yang sudah dipreprocess (.pkl)
with open(os.path.join(CURRENT_DIR, 'processed_artikel.pkl'), 'rb') as f:
    processed_paper = pickle.load(f)

# Load data asli dari Excel untuk menampilkan Judul, Tanggal, Isi, dan Link
paper_x = pd.read_excel(os.path.join(CURRENT_DIR, 'hasil_scraping_artikel.xlsx'))
paper = paper_x.values.tolist()

# Tentukan indeks kolom sesuai data Excel Anda
IDX_LINK    = 1   
IDX_JUDUL   = 2
IDX_TANGGAL = 3
IDX_ISI     = 4

# --- 3. FUNGSI PENCARIAN TF-IDF & COSINE SIMILARITY ---
def search_articles(query_raw, top_n=5):
    # Preprocessing Query (Sama persis dengan sistem di Colab Anda)
    query = query_raw.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    query = query.translate(remove_punctuation_map)
    query = stopword.remove(query)
    query = query.split()
    query = [stemmer.stem(w) for w in query]

    if not query:
        return []

    # Hitung TF-IDF & Cosine Similarity
    vectorizer2 = TfidfVectorizer(use_idf=True)
    corpus = [' '.join(query)] + processed_paper
    tfidf_matrix = vectorizer2.fit_transform(corpus)
    
    # Similarity baris pertama (query) dengan baris sisanya (dokumen-dokumen)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_idx = np.argsort(-scores)

    results = []
    seen = set()
    for i in ranked_idx:
        if scores[i] <= 0.0:
            break
        if i not in seen:
            seen.add(i)
            
            # Ambil cuplikan isi (preview) maksimal 250 karakter
            isi_full = str(paper[i][IDX_ISI])
            isi_preview = isi_full[:250] + '...' if len(isi_full) > 250 else isi_full
            
            results.append({
                'rank'   : len(results) + 1,
                'score'  : round(float(scores[i]), 4),
                'judul'  : paper[i][IDX_JUDUL],
                'tanggal': paper[i][IDX_TANGGAL],
                'link'   : paper[i][IDX_LINK],
                'isi'    : isi_preview,
            })
        if len(results) >= top_n:
            break
    return results

# --- 4. ROUTE FLASK ---
@app.route('/')
def home():
    query = request.args.get('q', '')
    limit = request.args.get('limit', 5, type=int) # Default menampilkan 5 hasil
    results = []
    
    if query:
        results = search_articles(query, top_n=limit)
        
    return render_template('index.html', query=query, results=results, limit=limit)

# Handler untuk Vercel Serverless
app.wsgi_app = app.wsgi_app