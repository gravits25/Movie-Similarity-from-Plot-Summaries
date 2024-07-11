from flask import Flask, request, jsonify
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load preprocessed data and models
movies_df = pd.read_csv('movies.csv')
movies_df['combined_plot'] = movies_df['wiki_plot'].astype(str) + " " + movies_df['imdb_plot'].astype(str)
movies_df['cleaned_plot_summary'] = movies_df['combined_plot'].apply(preprocess_text)

# Initialize TF-IDF Vectorizer and calculate similarity matrix
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['cleaned_plot_summary'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_similar_movies(title, top_n=10):
    if title not in movies_df['title'].values:
        return f"Movie titled '{title}' not found in the dataset."
    
    # Get the index of the movie
    movie_idx = movies_df[movies_df['title'] == title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top n similar movies
    sim_scores = sim_scores[1:top_n+1]
    similar_movies = [(movies_df.iloc[i[0]]['title'], i[1]) for i in sim_scores]
    
    return similar_movies

@app.route('/similar_movies', methods=['GET'])
def similar_movies():
    title = request.args.get('title')
    top_n = int(request.args.get('top_n', 10))
    similar_movies_list = get_similar_movies(title, top_n)
    return jsonify(similar_movies_list)

if __name__ == '__main__':
    app.run(debug=True)
