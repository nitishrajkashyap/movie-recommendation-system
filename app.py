import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.caption("Content-based filtering using TF-IDF & Cosine Similarity")

@st.cache_data
def get_data():
    raw_data = {
        'title': [
            'The Dark Knight', 'Inception', 'Interstellar', 'The Avengers', 
            'The Prestige', 'Iron Man', 'Spider-Man: Into the Spider-Verse', 
            'Man of Steel', 'The Matrix', 'Shutter Island'
        ],
        'genre': [
            'Action Crime Drama Thriller Superhero Batman Joker', 
            'Action Sci-Fi Thriller Mind-bending Dreams Inception', 
            'Adventure Drama Sci-Fi Space Wormhole Black Hole Time Travel', 
            'Action Sci-Fi Superhero Marvel Avengers Loki Earth', 
            'Drama Mystery Sci-Fi Magic Illusionists Rivalry', 
            'Action Sci-Fi Superhero Marvel Tony Stark Armor', 
            'Animation Action Adventure Superhero Miles Morales Multiverse', 
            'Action Adventure Sci-Fi Superhero Superman Alien', 
            'Action Sci-Fi Simulation Virtual Reality Cyberpunk AI', 
            'Mystery Thriller Psychological Asylum Plot Twist'
        ],
        'poster': [
            'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
            'https://image.tmdb.org/t/p/w500/oYiMoLZIy2B2E60jYQ5aP0g39wX.jpg',
            'https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg',
            'https://image.tmdb.org/t/p/w500/RYMX2wcKSpAr242RGlB5pW6zN1.jpg',
            'https://image.tmdb.org/t/p/w500/5vY2182mFFiRZChJ22hzMvT29Te.jpg',
            'https://image.tmdb.org/t/p/w500/78lPtwv72eTNqFW9COBYI0dZCZD.jpg',
            'https://image.tmdb.org/t/p/w500/iiZZdoQHeeFGivVqCDicZ1lGYw1.jpg',
            'https://image.tmdb.org/t/p/w500/8c4a8kE7PjhNJYJY8v8exWqM2DB.jpg',
            'https://image.tmdb.org/t/p/w500/f89U3w9nLviwsnhMuFLje8fZ2Ut.jpg',
            'https://image.tmdb.org/t/p/w500/4GDy0PHYX3VRXUt7x5ysL3zXpWh.jpg'
        ]
    }
    return pd.DataFrame(raw_data)

df = get_data()

# Vectorization & Similarity Matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['genre'])
sim_matrix = cosine_similarity(tfidf_matrix)

# Sidebar info
with st.sidebar:
    st.header("Project Info")
    st.write("Developer: **Nitish Raj**")
    st.write("Tech Stack: Python, Pandas, Scikit-learn, Streamlit")
    st.markdown("---")
    st.caption("Built for ML Internship Portfolio")

# UI Logic
st.markdown("---")
selected_movie = st.selectbox("Select a movie:", df['title'].values)

if st.button("Get Recommendations"):
    idx = df[df['title'] == selected_movie].index[0]
    
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:5]
    
    st.subheader(f"Movies similar to '{selected_movie}':")
    
    cols = st.columns(4)
    for col, (movie_idx, score) in zip(cols, scores):
        movie = df.iloc[movie_idx]
        with col:
            st.image(movie['poster'], caption=movie['title'])