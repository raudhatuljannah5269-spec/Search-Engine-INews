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

# 1. INISIALISASI SASTRAWI
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

# 2. SEBAIKNYA LOAD DATA SECARA AMAN
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
processed_pkl_path = os.path.join(CURRENT_DIR, 'processed_artikel.pkl')
excel_path = os.path.join(CURRENT_DIR, 'hasil_scraping_artikel.xlsx')

processed_paper = []
paper = []

# Cek & Load file .pkl
if os.path.exists(processed_pkl_path):
    with open(processed_pkl_path, 'rb') as f:
        processed_paper = pickle.load(f)
else:
    print(f"⚠️ Peringatan: File {processed_pkl_path} tidak ditemukan!")

# Cek & Load file .xlsx
if os.path.exists(excel_path):
    paper_x = pd.read_excel(excel_path)
    paper = paper_x.values.tolist()
else:
    print(f"⚠️ Peringatan: File {excel_path} tidak ditemukan!")

# Index kolom sesuai data Colab kamu
IDX_LINK    = 1   
IDX_JUDUL   = 2
IDX_TANGGAL = 3
IDX_ISI     = 4

# 3. FUNGSI LOGIKA PENCARIAN (DARI COLAB KAMU)
def search_articles(query_raw, top_n=5):
    if not processed_paper or not paper:
        return []

    query = query_raw.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    query = query.translate(remove_punctuation_map)
    query = stopword.remove(query)
    query = query.split()
    query = [stemmer.stem(w) for w in query]

    if not query:
        return []

    vectorizer2 = TfidfVectorizer(use_idf=True)
    corpus = [' '.join(query)] + processed_paper
    tfidf_matrix = vectorizer2.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_idx = np.argsort(-scores)

    results = []
    seen = set()
    for i in ranked_idx:
        if scores[i] <= 0.0:
            break
        if i not in seen:
            seen.add(i)
            
            # Batasi isi pratinjau teks artikel agar rapi
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

# 4. PATH ROUTING WEB FLASK
@app.route('/')
def home():
    query = request.args.get('q', '')
    limit = request.args.get('limit', 5, type=int)
    results = []
    
    if query:
        results = search_articles(query, top_n=limit)
        
    return render_template('index.html', query=query, results=results, limit=limit)

# Diperlukan untuk Vercel Serverless
app.wsgi_app = app.wsgi_app

# PERINTAH UTAMA UNTUK MENYALAKAN SERVER LOKAL DI WINDOWS
if __name__ == '__main__':
    print("Menyalakan server lokal...")
    app.run(debug=True, port=5000)