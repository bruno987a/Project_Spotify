import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors #Machine Learning algorithm @Lorenz
from sklearn.preprocessing import StandardScaler

# Cookie manager
try:
    from streamlit_cookies_manager import EncryptedCookieManager
except Exception:
    EncryptedCookieManager = None

# Initialize the EncryptedCookieManager (ensure to pip install streamlit-cookies-manager)
# Set an environment variable COOKIES_PASSWORD in production with a strong secret.
COOKIES_PASSWORD = os.environ.get("COOKIES_PASSWORD", "dev-secret-please-change")
cookies = None
if EncryptedCookieManager is not None:
    cookies = EncryptedCookieManager(prefix="project_spotify/", password=COOKIES_PASSWORD)
    if not cookies.ready():
        # Wait for cookies component to initialize and load values
        st.stop()

candidate_songs = []

# -------------------------
# Streamlit app setup
# -------------------------
st.set_page_config(page_title="Smart Playlist Generator", page_icon="🎧", layout="wide")

st.title("Smart Playlist Generator")
st.markdown("Create personalized playlists based on your musical preferences and feedback.")

# Helper to load cookie value safely
def cookie_get(key, default=None):
    if cookies is None:
        return default
    val = cookies.get(key)
    return val if val is not None else default

def cookie_set(key, val):
    if cookies is None:
        return
    # Store JSON-serializable content as JSON string for safety
    try:
        cookies[key] = val
    except Exception:
        # fallback to JSON string
        cookies[key] = json.dumps(val)

def cookie_delete(key):
    if cookies is None:
        return
    try:
        del cookies[key]
    except Exception:
        pass

def cookie_save():
    if cookies is None:
        return
    try:
        cookies.save()
    except Exception:
        pass

# Initialize session state for progress tracking (load from cookies if present)
if "step" not in st.session_state:
    step_cookie = cookie_get("step")
    if step_cookie is not None:
        try:
            st.session_state.step = int(step_cookie)
        except Exception:
            st.session_state.step = 2
    else:
        st.session_state.step = 2

if "ratings" not in st.session_state:
    ratings_cookie = cookie_get("ratings")
    if ratings_cookie:
        # ratings might be stored as JSON string
        try:
            st.session_state.ratings = json.loads(ratings_cookie) if isinstance(ratings_cookie, str) else ratings_cookie
        except Exception:
            st.session_state.ratings = {}
    else:
        st.session_state.ratings = {}

if "current_user" not in st.session_state:
    cu = cookie_get("current_user")
    st.session_state.current_user = cu if cu is not None else "User 1"

if "criteria_confirmed" not in st.session_state:
    cc = cookie_get("criteria_confirmed")
    st.session_state.criteria_confirmed = bool(cc) if cc is not None else False

if "evaluation_done" not in st.session_state:
    ed = cookie_get("evaluation_done")
    st.session_state.evaluation_done = bool(ed) if ed is not None else False

# Also restore chosen_genre, n_desired_songs, num_users if present
if "chosen_genre" not in st.session_state:
    cg = cookie_get("chosen_genre")
    st.session_state.chosen_genre = int(cg) if cg is not None else None

if "n_desired_songs" not in st.session_state:
    nds = cookie_get("n_desired_songs")
    st.session_state.n_desired_songs = int(nds) if nds is not None else 15

if "num_users" not in st.session_state:
    nu = cookie_get("num_users")
    st.session_state.num_users = int(nu) if nu is not None else 1

if "current_user_idx" not in st.session_state:
    idx_cookie = cookie_get("current_user_idx")
    st.session_state.current_user_idx = int(idx_cookie) if idx_cookie is not None else 0

if "candidate_songs" not in st.session_state:
    st.session_state.candidate_songs = None

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

    # If we already had a chosen genre in cookies, pre-select it
    key_genre = st.selectbox("Select Genre:", list(genre_map.keys()))
    chosen_genre = genre_map[key_genre]
    n_desired_songs = st.slider("Select desired playlist length (songs):", 5, 30, 15)                  #the user choses the number of recommended songs

    num_users = st.slider("How many people will rate?", 1, 5, 1) 

    # button for continuing the workflow and start the rating process   
    if st.button("Confirm and Continue"):                                                                    
        st.session_state.criteria_confirmed = True
        st.session_state.step = 3
        
        #store parameters in session_state
        st.session_state.chosen_genre = chosen_genre
        st.session_state.n_desired_songs = n_desired_songs
        st.session_state.num_users = num_users

        st.session_state.current_user_idx = 0

        # Save to cookies (persist preferences)
        cookie_set("chosen_genre", st.session_state.chosen_genre)
        cookie_set("n_desired_songs", st.session_state.n_desired_songs)
        cookie_set("num_users", st.session_state.num_users)
        cookie_set("criteria_confirmed", True)
        cookie_set("step", st.session_state.step)
        cookie_save()
    
        st.success("Preferences saved. Proceed to Quick Evaluation.")

# -------------------------
# STEP 3 — Quick Evaluation
# -------------------------
if st.session_state.step >= 3 and st.session_state.criteria_confirmed:  
    st.header("Step 2 – Quick song evaluation")
    st.write("Please rate the following songs:")

    num_users = st.session_state.num_users
    current_idx = st.session_state.get("current_user_idx", 0)

    st.caption(f"Rater {current_idx + 1} of {num_users}")

    # --- Name of current user ---
    default_name = f"User {current_idx + 1}"
    name_key = f"user_name_{current_idx}"


    #Who is rating
    current_user = st.text_input(
        "Who is rating?", value=st.session_state.current_user,
        help="Enter your name here",
        key=name_key,
    )
    
    #Fallback, if someone leaves it empty
    if not current_user.strip():
        current_user = default_name

    st.session_state.current_user = current_user
    # Persist last current_user to cookie
    cookie_set("current_user", st.session_state.current_user)
    cookie_save()

    #rating-dict for current user
    if current_user not in st.session_state.ratings:
        st.session_state.ratings[current_user] = {}
    user_ratings = st.session_state.ratings[current_user]
    
   
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

    # same songs for everyone
    if st.session_state.candidate_songs is None: 
        st.session_state.candidate_songs = rand_track_genre(st.session_state.chosen_genre, 5) # hier noch auswahl der anzahl songs ermöglichen evtl.

    songs_df = st.session_state.candidate_songs

    for idx, (track_id, row) in enumerate(songs_df.iterrows()):
        c1, c2, c3 = st.columns([4, 4, 3])

        c1.write(row["title"])
        c2.write(row["artist"])

        rating = c3.slider(
            label="", 
            min_value=1,
            max_value=5,
            value=3,
            key=f"rating_{idx}",
            label_visibility="collapsed",
            step=1,
        )

        #save the rating for this user
        user_ratings[row["track_id"]] = rating

    st.session_state.ratings[current_user] = user_ratings

    # Persist ratings and index to cookies periodically
    cookie_set("ratings", json.dumps(st.session_state.ratings))
    cookie_set("current_user_idx", st.session_state.current_user_idx)
    cookie_set("step", st.session_state.step)
    cookie_save()

    if current_idx < num_users - 1: 
        if st.button("Next person"):
            st.session_state.current_user_idx = current_idx + 1 
            # persist index
            cookie_set("current_user_idx", st.session_state.current_user_idx)
            cookie_save()
            st.experimental_rerun()
    
    else: 
        if st.button("Generate final playlist"):
            st.session_state.evaluation_done = True
            st.session_state.step = 4
            cookie_set("evaluation_done", True)
            cookie_set("step", st.session_state.step)
            cookie_save()
            st.experimental_rerun()
    
    
    # ------------------------------
    # START MACHINE LEARNING PART
    # ------------------------------
    
    # Vector definition with computed features
    # The ML block will only run once evaluation_done is set and step advanced; keep it here for clarity.
    if st.session_state.evaluation_done and st.session_state.step >= 4:
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

        #username to track id: rating, continue if not rated 
        for username, rating_dict in st.session_state.ratings.items():
            if not rating_dict:
                continue 

            #Track-IDs and Ratings of this user
            rated_track_ids = list(rating_dict.keys())
       
            #only use tracks, for wich we have features
            rated_track_ids = [tid for tid in rated_track_ids if str(tid) in features_14_scaled.index or tid in features_14_scaled.index]
            # ensure types align (index may be string or int)
            rated_track_ids = [tid if tid in features_14_scaled.index else str(tid) for tid in rated_track_ids]
            ratings_list = [rating_dict[tid] for tid in rated_track_ids]

            if len(rated_track_ids) == 0:
                continue

            profile = build_user_profile(ratings_list, rated_track_ids, features_14_scaled)
            user_profiles.append(profile)

        if len(user_profiles) == 0:
            st.error("There are no ratings - no recommendation possible.")
            st.stop()
        
        # Group vector representing music taste = average of user profiles
        group_profile = np.mean(user_profiles, axis=0)   # Shape: (14,)
    
        # Adjustment instruments for emphasising certain features
        group_vector = group_profile.copy()
    
        # give more weight to one feature  (e.g. factor 1.5)
        feature_name_to_boost = "rmse_01_mean"   # <- change as desired
        if feature_name_to_boost in feature_cols:
            idx = feature_cols.index(feature_name_to_boost)
            group_vector[idx] *= 1.5
    
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
        recommended_ids = recommend(group_vector, st.session_state.n_desired_songs).tolist()
        st.session_state.recommended_ids = recommended_ids

        # persist recommendations
        cookie_set("recommended_ids", json.dumps(recommended_ids))
        cookie_save()

# -------------------------
# END MACHINE LEARNING
# -------------------------
# STEP 4 — Final Playlist
if st.session_state.step >= 4 and st.session_state.evaluation_done:
    st.header("Step 3 – Your final recommended playlist")
    st.write("Generated based on your preferences and evaluations:")

    t = pd.read_csv("data/tracks_small.csv")
    s_t_simple = t[["track_id", "title", "artist"]]

    recommended_ids = st.session_state.get("recommended_ids", [])
    df_final = s_t_simple[s_t_simple["track_id"].isin(recommended_ids)][["title", "artist"]]

    st.dataframe(df_final, use_container_width=True)

    st.markdown("**Summary:**")
    st.write(f"- Total songs: {len(df_final)}")

    if st.button("Start Over"):
        st.session_state.step = 2
        st.session_state.ratings = {}
        st.session_state.criteria_confirmed = False
        st.session_state.evaluation_done = False
        # clear cookies related to progress
        cookie_delete("ratings")
        cookie_delete("chosen_genre")
        cookie_delete("n_desired_songs")
        cookie_delete("num_users")
        cookie_delete("step")
        cookie_delete("criteria_confirmed")
        cookie_delete("evaluation_done")
        cookie_save()
        st.experimental_rerun()

    st.button("Save Playlist to Spotify (coming soon)")

    # Option to clear cookies manually
    if cookies is not None:
        if st.button("Clear saved cookies"):
            # Remove all cookie keys set by this app
            for k in list(dict(cookies).keys()):
                try:
                    del cookies[k]
                except Exception:
                    pass
            cookie_save()
            st.success("Saved cookies cleared. Refreshing...")
            st.experimental_rerun()