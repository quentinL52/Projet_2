import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.stem.porter import PorterStemmer
st.set_page_config(layout="wide")

################# data set #############################################################################

df2 = pd.read_csv('Reco_vecto.csv')
df_norm = pd.read_csv("DF_KNN.csv")

############################# modele de ML Keyword #############################################################

cv = TfidfVectorizer(stop_words='english', token_pattern=r"(?u)\b[a-zA-Z0_9]+\b")
vectors = cv.fit_transform(df2['mots']).toarray()
similarity = cosine_similarity(vectors)

def recommend(query, df2):
    query = query.lower()
    movie_index = df2[df2['title'].str.lower() == query].index
    if len(movie_index) > 0:
        movie_index = movie_index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        movie_rec = [df2.iloc[i[0]]['title'] for i in movies_list]
    else:
        movie_rec = df2[df2['mots'].str.contains(query, case=False, na=False)]['title'].head(5).tolist()
    return movie_rec

############################# modele de ML KNN #############################################################
# colonnes utilisées pour la recherche
list_col = ['averageRating',
            'popularity',
            'startYear',
            'runtimeMinutes',
            "Western",
            "Thriller",
            "Musical",
            "Romance",
            "Documentary"
            ]

# Model NearestNeighbors
X_norm = df_norm[list_col]
movie_reco = NearestNeighbors(n_neighbors=6).fit(X_norm)

def reco_movie (letitre) :
    try :
        id_input = df_norm[df_norm.title == letitre].index[0]

        # Définition de X_reco
        X_reco = [X_norm.iloc[id_input, :].values]

        # Utilisation de "kneighbors" pour obtenir la distance et les indices des recommandations
        distance , indices = movie_reco.kneighbors(X_reco)

        indices = indices[0,1:]  #<---- Modifier à [0,1:] / [0,0:] pour masquer / afficher les infos du film que l'on a utilisé pour la recherche

        df_reco = df2.iloc[indices,:]
        return df_reco
    except :
         pass

########################## banniere avec logo et titre ###################################################

col1, col2 = st.columns(2)
with col1:
    st.image("logo.png")

with col2:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.write("""
 <head>
    <title>
        Center text horizontally and
      vertically inside a div block
    </title>
    <style>
        .container {
            height: 200px;
            width: 400px;
            position: relative;
            border: 1px solid black;
            border-radius: 40px;
        }

        h1 {
            position: absolute;
            top: 50%;
            color: black;
            left: 50%;
            transform: translate(-50%, -50%);
            justify-content: center;
            align-items: center;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Bienvenue sur Creuxciné</h1>
    </div>
</body>

</html>
        """, unsafe_allow_html=True)
    
############################################# barre de selection des films #################################

option = st.selectbox("Choisissez un film pour obtenir une recommandation", df2['title'].unique())
selected_movie = None
selected_movie2 = None

###################################### popover descriptif du film choisi ###################################

if option:
    movie_row = df2[df2['title'] == option].iloc[0]
    movie_filter = df_norm[df_norm["title"] == option]  
    movie_image_path = movie_row['poster_path']
    movie_description = movie_row['overview']
    movie_duration = movie_row['runtimeMinutes']
    movie_rating = movie_row['averageRating']
    movie_genre = movie_row['genres']
    movie_actors = movie_row['Actors_names']
    movie_year = movie_row['startYear']

    actor_list = movie_actors.split(',')[:5]
    formatted_actors = ', '.join(actor_list)

    with st.popover(f"Plus d'infos sur {option}"):
        st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 10px;">
                <div style="display: flex;">
                    <div style="flex: 1; padding-right: 20px;">
                        <img src="{movie_image_path}" alt="{option}" style="width: 100%; border-radius: 10px;">
                    </div>
                    <div style="flex: 2;">
                        <h2 style='font-size: 40px;'>{option}</h2><br>
                        <strong>Description:</strong> {movie_description}<br>
                        <strong>Durée:</strong> {movie_duration} min<br>
                        <strong>Note:</strong> {movie_rating}/10<br>
                        <strong>Genre:</strong> {movie_genre}<br>
                        <strong>Année:</strong> {movie_year}<br>
                        <strong>Acteurs:</strong> {formatted_actors}<br>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

############################### lignes des films recommandé vecto ###########################################################
 
    movie_recommendation = recommend(option,df2)
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
                movie_actors = movie_row.iloc[0]['Actors_names']
                movie_genre = movie_row.iloc[0]['genres']
                movie_year = movie_row.iloc[0]['startYear']

############################# bouton pour les infos de films en dessous des images ###################################

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
            movie_year = movie_row['startYear']

            actor_list = movie_actors.split(',')[:5]
            formatted_actors = ', '.join(actor_list)

            st.markdown(f"""
                <div id="plus d'infos" style="border: 1px solid #ddd; padding: 10px; border-radius: 10px; margin-top: 20px;">
                    <div style="display: flex;">
                        <div style="flex: 1; padding-right: 20px;">
                            <img src="{movie_image_path}" alt="{selected_movie}" style="width: 100%; border-radius: 10px;">
                        </div>
                        <div style="flex: 2;">
                            <h2 style='font-size: 40px;'>{selected_movie}</h2><br>
                            <strong>Description:</strong> {movie_description}<br>
                            <strong>Durée:</strong> {movie_duration} min<br>
                            <strong>Année:</strong> {movie_year}<br>
                            <strong>Note:</strong> {movie_rating}/10<br>
                            <strong>Genre:</strong> {movie_genre}<br>
                            <strong>Acteurs:</strong> {formatted_actors}<br>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

############################### lignes des films recommandé KNN ###########################################################
 
    movie_recommendation2 = reco_movie(option)["title"].tolist()

    if movie_recommendation2:
        
        st.subheader("Voici d'autres films similaires à votre recherche :")
        num_cols = 5
        cols = st.columns(num_cols)

        for i2, movie2 in enumerate(movie_recommendation2):
            movie_row2 = df2[df2['title'] == movie2]
            if not movie_row2.empty:
                movie_image_path = movie_row2.iloc[0]['poster_path']
                movie_description = movie_row2.iloc[0]['overview']
                movie_duration = movie_row2.iloc[0]['runtimeMinutes']
                movie_rating = movie_row2.iloc[0]['averageRating']
                movie_actors = movie_row2.iloc[0]['Actors_names']
                movie_genre = movie_row2.iloc[0]['genres']
                movie_year = movie_row2.iloc[0]['startYear']

############################# bouton pour les infos de films en dessous des images ###################################

                with cols[i2 % num_cols]:
                    st.image(movie_image_path, use_column_width=True)
                    if st.button("Plus d'infos", key=f"button_{i2+6}"):
                        selected_movie2 = movie2

        if selected_movie2:
            movie_row2 = df2[df2['title'] == selected_movie2].iloc[0]
            movie_image_path = movie_row2['poster_path'] 
            movie_description = movie_row2['overview']
            movie_duration = movie_row2['runtimeMinutes']
            movie_rating = movie_row2['averageRating']
            movie_genre = movie_row2['genres']
            movie_year = movie_row2['startYear']

            actor_list = movie_actors.split(',')[:5]
            formatted_actors = ', '.join(actor_list)

            st.markdown(f"""
                <div id="plus d'infos" style="border: 1px solid #ddd; padding: 10px; border-radius: 10px; margin-top: 20px;">
                    <div style="display: flex;">
                        <div style="flex: 1; padding-right: 20px;">
                            <img src="{movie_image_path}" alt="{selected_movie2}" style="width: 100%; border-radius: 10px;">
                        </div>
                        <div style="flex: 2;">
                            <h2 style='font-size: 40px;'>{selected_movie2}</h2><br>
                            <strong>Description:</strong> {movie_description}<br>
                            <strong>Durée:</strong> {movie_duration} min<br>
                            <strong>Année:</strong> {movie_year}<br>
                            <strong>Note:</strong> {movie_rating}/10<br>
                            <strong>Genre:</strong> {movie_genre}<br>
                            <strong>Acteurs:</strong> {formatted_actors}<br>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)