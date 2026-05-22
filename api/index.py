from flask import Flask, render_template, request
import requests

app = Flask(__name__, template_folder='../templates')

def ambil_data_pencarian(query):
    # Menggunakan API gratis DuckDuckGo untuk mengambil hasil pencarian secara real-time
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        
        hasil_pencarian = []
        
        # Ekstrak data dari topik terkait yang diberikan API
        if "RelatedTopics" in data:
            for topic in data["RelatedTopics"]:
                if "Text" in topic and "FirstURL" in topic:
                    hasil_pencarian.append({
                        "title": topic["Text"].split(" - ")[0],
                        "snippet": topic["Text"],
                        "url": topic["FirstURL"]
                    })
        return hasil_pencarian
    except Exception as e:
        print(f"Gagal mengambil data: {e}")
        return []

@app.route('/')
def home():
    # Mengambil kata kunci yang diketik user di form pencarian
    query = request.args.get('q', '')
    results = []
    
    if query:
        results = ambil_data_pencarian(query)
        
    # Mengirimkan data kata kunci dan hasil pencarian ke file HTML
    return render_template('index.html', query=query, results=results)

# Bagian penting agar dikenali oleh sistem serverless Vercel
app.wsgi_app = app.wsgi_app