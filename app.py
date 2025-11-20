import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors #Machine Learning algorithm @Lorenz
from sklearn.preprocessing import StandardScaler

# -------------------------
# 
# -------------------------

candidate_songs = []

# -------------------------
# Streamlit app setup
# -------------------------
st.set_page_config(page_title="Smart Playlist Generator", page_icon="🎧", layout="wide")

st.title("Smart Playlist Generator")
st.markdown("Create personalized playlists based on your musical preferences and feedback.")

# Initialize session state for progress tracking
if "step" not in st.session_state:
    st.session_state.step = 2
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "criteria_confirmed" not in st.session_state:
    st.session_state.criteria_confirmed = False
if "evaluation_done" not in st.session_state:
    st.session_state.evaluation_done = False

# -------------------------
# STEP 1 — Import Playlist
# -------------------------
#deleted

# -------------------------
# STEP 2 — Generation Criteria
# -------------------------
if st.session_state.step >= 2:
    st.header("Step 1 – Playlist generation criteria")
    similarity = st.selectbox("Select similarity level:",
    ["None", "Genre", "Artist", "Mixed"],
    index=0,  # default selection is "None"
    format_func=lambda x: f"*{x}*" if x=="None" else x,
    key="similarity")

# Song selecetion for rating 
    genre_map = {"Rock/Metal/Punk": 1, "Pop/Synth": 2, "Electronic/IDM": 3, "Hip-Hop/RnB": 4,    
    "Jazz/Blues": 5, "Classical": 6, "Folk/Country/Americana": 7, "World/Reggae/Latin": 8,
    "Experimental/Sound Art": 9, "Spoken/Soundtrack/Misc": 10, "Funk": 11}   

    key_genre = st.selectbox("Select Genre:", list(genre_map.keys()))                                  #the user choses his genre he wishes, recommendations for 
    chosen_genre = genre_map[key_genre]                                                                #the selected genre gets maped to the associated number 
    n_desired_songs = st.slider("Select desired playlist length (songs):", 5, 30, 15)                  #the user choses the number of recommended songs


# button for continuing the workflow and start the rating process    
    if st.button("Confirm and Continue"):                                                                    
        st.session_state.criteria_confirmed = True
        st.session_state.step = 3
        st.success("Preferences saved. Proceed to Quick Evaluation.")


# -------------------------
# STEP 3 — Quick Evaluation
# -------------------------
if st.session_state.step >= 3 and st.session_state.criteria_confirmed:  
    st.header("Step 3 – Quick song evaluation")
    st.write("Please rate the following songs:")

    from ast import literal_eval
    from random import choice
    
    gmi = pd.read_csv("data/genre_with_main_identity.csv")                              #reading the data for the genres
    s_genres = gmi[["genre_id", "main_category_id"]]                                    #the gmi gets reduced to two data points

    t = pd.read_csv("data/tracks_small.csv")  #importing the data for the tracks
    s_t = pd.DataFrame({"track_id": t["track_id"], "genres_all": t["genres_all"].fillna("[]").apply(literal_eval), "title": t["title"], "artist": t["artist"]}) 

    def rand_track_genre(main_cat_id, n):
        genre_ids = list(set(s_genres.loc[s_genres["main_category_id"] == main_cat_id, "genre_id"]))
    
        rand_gen_l = [choice(genre_ids) for dig in range(n)]
        
        p_to_rate = []
    
        for g_id in rand_gen_l:
            poss_songs = s_t[s_t["genres_all"].apply(lambda ids: g_id in ids)]
            p_to_rate.append(poss_songs.sample(1))
            to_rate = pd.concat(p_to_rate)
        return to_rate

    # Display songs with rating buttons
    if "candidate_songs" not in st.session_state: 
        st.session_state.candidate_songs = rand_track_genre(chosen_genre, 5) # hier noch auswahl der anzahl songs ermöglichen evtl.

    songs_df = st.session_state.candidate_songs
    
    for idx, (track_id, row) in enumerate(songs_df.iterrows()):
        cols = st.columns([3, 3, 2, 2, 2])
        
        cols[0].write(row["title"])
        cols[1].write(row["artist"])
        
    #rating process @Loris
        
        rating = cols[4].radio(
            "Rating ", 
            options=[1,2,3,4,5],
            format_func=lambda x: "⭐" * x,
            horizontal=True, 
            key=f"song_{idx}") 
        st.session_state.ratings[row["track_id"]] = rating

    
if st.session_state.step >= 3 and st.session_state.criteria_confirmed:
    if st.button("Generate Final Playlist"):
        st.session_state.evaluation_done = True
        st.session_state.step = 4
        st.success("Evaluation submitted! Proceed to Final Playlist.")
    
    
    # ------------------------------
    # START MACHINE LEARNING PART
    # ------------------------------
    
        # Vector definition with computed features
        
        features = pd.read_csv("data/reduced_features.csv", index_col=0)  # track_id as index
        
        feature_cols = [
            "mfcc_01_mean", "mfcc_02_mean", "mfcc_03_mean", "mfcc_04_mean", "mfcc_05_mean",
            "mfcc_06_mean", "mfcc_07_mean", "mfcc_08_mean", "mfcc_09_mean", "mfcc_10_mean",
            "rmse_01_mean",
            "spectral_centroid_01_mean",
            "spectral_bandwidth_01_mean",
            "chroma_var"
        ]
        features_14 = features[feature_cols].copy()
        
        scaler = StandardScaler()
        X_14 = scaler.fit_transform(features_14)
        
        features_14_scaled = pd.DataFrame(X_14, index=features.index, columns=feature_cols)
        
        # Nearest Neighbours setup and function call
        
        """
        # Training of the model, can be activated if necessary.
        # features_14_scaled: DataFrame (index = track_id, columns = feature_cols)
        X = features_14_scaled.values                     # NumPy-Matrix (n_tracks, 14)
        track_ids = features_14_scaled.index.to_numpy()   # Track-IDs passend zu X
        
        knn_model = NearestNeighbors(
            n_neighbors=200,      # erstmal „viele“, filtern später runter
            metric="cosine"
        )
        knn_model.fit(X)
        """
        
        # Rated songs in form of a list in the same order as ratings, ist falsch geratete Liste ist ein dict!!! zugeordnet zur track_id!!!
        # Annahme: Songs zur Bewertung in chronologischer Abfolge unter songs_df abgespeichert. sollte stimmen
        rated_track_ids = list(st.session_state.ratings.keys())
        
        # Ratings from streamlit per user
        ratings_user1 = [
            st.session_state.ratings[track_id]
            for track_id in rated_track_ids
        ]# numbers from 1-5, as a list for each song 
        #ratings_user2 = rating.user2   # activate them
        #ratings_user3 = rating.user3   # @Loris vielleicht noch Name anpassen damits deine Zahlen übernimmt
        #ratings_user4 = rating.user4
        #ratings_user5 = rating.user5
        
        # dictionary of "user" - rating pairs
        user_ratings = {
            "user1": ratings_user1,
         #   "user2": ratings_user2,
         #   "user3": ratings_user3,
         #   "user4": ratings_user4,
         #   "user5": ratings_user5
            # add more users if necessary
            }
        
        # define function to create seed vector per user
        
        def build_user_profile(ratings_list, rated_track_ids, features_14_scaled):
            """
            ratings_list: Liste von Ratings (1–5), gleiche Reihenfolge wie rated_track_ids
            rated_track_ids: Liste der track_ids aus songs_df
            features_df: features_14_scaled (index = track_id), muss ev. noch assigned werden
            """
        
            # Convert ratings to Numpy arrays
            ratings = np.asarray(ratings_list, dtype=float)
        
            # Set vectors of rated songs
            vecs = features_14_scaled.loc[rated_track_ids].values          # Shape: (n_rated, 14)
        
            # Weighted Average (Ratings = weights)
            profile_vector = np.average(vecs, axis=0, weights=ratings)
        
            return profile_vector    # Shape: (14,)
        
        # Collect all seed vector of users into a list
        
        user_profiles = []
        
        for ratings_list in user_ratings.values():
            profile = build_user_profile(ratings_list, rated_track_ids, features_14_scaled)
            user_profiles.append(profile)
        
        # Group vector representing music taste = average of user profiles
        
        group_profile = np.mean(user_profiles, axis=0)   # Shape: (14,)
        
        # Adjustment instruments for emphazising certain features
        
        group_vector = group_profile.copy()
        
        # give more weight to one feature  (e.g. factor 1.5)
        feature_name_to_boost = "rmse_01_mean"   # <- change as desired
        if feature_name_to_boost in feature_cols:
            idx = feature_cols.index(feature_name_to_boost)
            group_vector[idx] *= 1.5
        
        # can add multiple of these blocks for more control over algorithm
        
        # --- kNN-Setup  ---
        
        X = features_14_scaled.values                     # Matrix (n_tracks, 14)
        track_ids = features_14_scaled.index.to_numpy()   # Track-IDs in the same order
        
        knn_model = NearestNeighbors(metric="cosine", n_neighbors=200)
        knn_model.fit(X)
        
        # Simple function design
        
        def recommend(group_vec, n_desired_songs):
            _, idx = knn_model.kneighbors(group_vec.reshape(1, -1), n_neighbors=n_desired_songs)
            return track_ids[idx[0]]
        
        # final function call
        recommended_ids = recommend(group_vector, n_desired_songs).tolist()

# -------------------------
# END MACHINE LEARNING
# -------------------------
# STEP 4 — Final Playlist
# -------------------------
if st.session_state.step >= 4 and st.session_state.evaluation_done:
    st.header("Step 4 – Your final recommended playlist")
    st.write("Generated based on your preferences and evaluations:")

    df_final = s_t[s_t["track_id"].isin(recommended_ids)][["title", "artist"]]
    st.dataframe(df_final, use_container_width=True)

    st.markdown("**Summary:**")
    st.write(f"- Total songs: {len(df_final)}")

    if st.button("Start Over"):
        st.session_state.step = 2
        st.session_state.ratings = {}
        st.session_state.criteria_confirmed = False
        st.session_state.evaluation_done = False
        st.experimental_rerun()

    st.button("Save Playlist to Spotify (coming soon)")

