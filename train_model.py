import pandas as pd

# Load the dataset
movies_df = pd.read_csv('movies.csv')

# Display the first few rows of the dataframe
print(movies_df.head())

import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
movies_df = pd.read_csv('movies.csv')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        text = ""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Fill NaN values with empty strings and combine the plot summaries
movies_df['wiki_plot'] = movies_df['wiki_plot'].fillna('')
movies_df['imdb_plot'] = movies_df['imdb_plot'].fillna('')
movies_df['combined_plot'] = movies_df['wiki_plot'] + " " + movies_df['imdb_plot']

# Apply preprocessing
movies_df['cleaned_plot_summary'] = movies_df['combined_plot'].apply(preprocess_text)

# Check the preprocessed data
print(movies_df[['title', 'cleaned_plot_summary']].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transform the cleaned plot summaries into TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['cleaned_plot_summary'])

from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity matrix
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

# Example usage
similar_movies = get_similar_movies("Star Wars", top_n=5)
print(similar_movies)