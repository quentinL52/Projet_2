import streamlit as st
import pandas as pd
import numpy as np
import time
import streamlit_shadcn_ui as ui
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
st.set_page_config(layout="wide")

df2 = pd.read_csv('df2.csv')

print(df2.isna().sum())

cv = TfidfVectorizer(stop_words='english', token_pattern=r"(?u)\b[a-zA-Z0_9]+\b")
vectors = cv.fit_transform(df2['mots']).toarray()
similarity = cosine_similarity(vectors)

def recommend(query, df2):
    query = query.lower()
    movie_index = df2[df2['title'] == query].index
    if len(movie_index) > 0:
        movie_index = movie_index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        movie_rec = [df2.iloc[i[0]]['title'] for i in movies_list]
    else:
        movie_rec = df2[df2['mots'].str.contains(query, case=False, na=False)]['title'].head(5).tolist()
    return movie_rec

col1, col2 = st.columns(2)
with col1:
    st.image("logo.png")

with col2:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    container = st.container()
    container.header('Bienvenue sur Creuxciné')

option = st.text_input("tapez un mot clef ou un film pour obtenir une recommandation")

if option:
    movie_recommendation = recommend(option, df2)
    if movie_recommendation:
        st.subheader("Voici des films similaires à votre recherche :")
        num_cols = 5
        cols = st.columns(num_cols)

        for i, movie in enumerate(movie_recommendation):
            movie_row = df2[df2['title'] == movie]
            if not movie_row.empty:
                movie_image_path = movie_row.iloc[0]['poster_path']
                movie_description = movie_row.iloc[0]['overview']
                movie_duration = movie_row.iloc[0]['runtimeMinutes']
                movie_rating = movie_row.iloc[0]['averageRating']
                movie_genre = movie_row.iloc[0]['genres']
                

                with cols[i % num_cols]:
                    st.image(movie_image_path, use_column_width=True)
                    if st.button("Plus d'infos", key=f"button_{i}"):
                        selected_movie = movie

        if selected_movie:
            movie_row = df2[df2['title'] == selected_movie].iloc[0]
            movie_image_path = movie_row['poster_path']
            movie_description = movie_row['overview']
            movie_duration = movie_row['runtimeMinutes']
            movie_rating = movie_row['averageRating']
            movie_genre = movie_row['genres']

            st.markdown(f"""
                <div id="plus d'infos" style="border: 1px solid #ddd; padding: 10px; border-radius: 10px; margin-top: 20px;">
                    <div style="display: flex;">
                        <div style="flex: 1; padding-right: 20px;">
                            <img src="{movie_image_path}" alt="{selected_movie}" style="width: 100%; border-radius: 10px;">
                        </div>
                        <div style="flex: 2;">
                            <h2 style='font-size: 24px;'>{selected_movie}</h2><br>
                            <strong>Description:</strong> {movie_description}<br>
                            <strong>Durée:</strong> {movie_duration} min<br>
                            <strong>Note:</strong> {movie_rating}/10<br>
                            <strong>Genre:</strong> {movie_genre}<br>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("Aucun film trouvé pour votre recherche.")
