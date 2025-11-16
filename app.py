import streamlit as st
import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.neighbors import NearestNeighbors #Machine Learning algorithm @Lorenz

# Vector definition with computed features

""" Define path for local and project data for test version.
    Can later be adapted for final version."""

LOCAL_DATA_DIR = Path("/Users/alexbuser/Library/Mobile Documents/com~apple~CloudDocs/02_Business/04_IT/01_Cobrazähmerei/10_Fundamentals of CS HSG/Group project/data") #local data on Alex's computer

tracks = pd.read_csv(LOCAL_DATA_DIR / "tracks.csv", index_col=0, low_memory=False)
features = pd.read_csv(LOCAL_DATA_DIR / "features.csv", index_col=0, low_memory=False)

BASE_DIR = Path(__path__).resolve().parent #path to parent file or our project
PROJECT_DATA_DIR = BASE_DIR / "data" #path to data samples in our project

genres = pd.read_csv(PROJECT_DATA_DIR / "genre_with_main_identity.csv")


# -------------------------
# 
# -------------------------
songs_data = [
    {"Title": "Spotify", "Artist": "Taylor Swift", "Genre": "Pop", "Duration": "3:55"},
    {"Title": "Do I Wanna Know?", "Artist": "Arctic Monkeys", "Genre": "Indie Rock", "Duration": "4:33"},
    {"Title": "Bad Guy", "Artist": "Billie Eilish", "Genre": "Pop", "Duration": "3:14"},
    {"Title": "Blinding Lights", "Artist": "The Weeknd", "Genre": "Synthpop", "Duration": "3:20"},
    {"Title": "Levitating", "Artist": "Dua Lipa", "Genre": "Pop", "Duration": "3:23"},
]

candidate_songs = []

# -------------------------
# Streamlit app setup
# -------------------------
st.set_page_config(page_title="Smart Playlist Generator", page_icon="🎧", layout="wide")

st.title("Smart Playlist Generator")
st.markdown("Create personalized playlists based on your musical preferences and feedback.")

# Initialize session state for progress tracking
if "step" not in st.session_state:
    st.session_state.step = 1
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
if "playlist_imported" not in st.session_state:
    st.session_state.playlist_imported = False
if "criteria_confirmed" not in st.session_state:
    st.session_state.criteria_confirmed = False
if "evaluation_done" not in st.session_state:
    st.session_state.evaluation_done = False

# -------------------------
# STEP 1 — Import Playlist
# -------------------------
if st.session_state.step >= 1:
    st.header("Step 1 – Import your Spotify playlist")
    playlist_id = st.text_input("Enter your Spotify Playlist ID or URL:", placeholder="e.g., https://open.spotify.com/playlist/...")

    if st.button("Import Playlist"):
        st.session_state.playlist_imported = True
        st.session_state.step = 2
        st.success("Playlist imported successfully (mock data shown below).")
        df = pd.DataFrame(songs_data)
        st.subheader("Your Playlist Preview")
        st.dataframe(df, use_container_width=True)

        st.markdown("**Summary:**")
        st.write("- Total songs: ", len(df))
        st.write("- Top genres: Pop, Indie Rock, Synthpop")
        st.write("- Top artists: Taylor Swift, Arctic Monkeys, Billie Eilish")


# -------------------------
# STEP 2 — Generation Criteria
# -------------------------
if st.session_state.step >= 2 and st.session_state.playlist_imported:
    st.header("Step 2 – Playlist generation criteria")
    similarity = st.selectbox("Select similarity level:",
    ["None", "Genre", "Artist", "Mixed"],
    index=0,  # default selection is "None"
    format_func=lambda x: f"*{x}*" if x=="None" else x,
    key="similarity")

# Song selecetion for rating 
    genre_map = {"Rock/Metal/Punk": 1, "Pop/Synth": 2, "Electronic/IDM": 3, "Hip-Hop/RnB/Funk": 4, 
    "Jazz/Blues": 5, "Classical": 6, "Folk/Country/Americana": 7, "World/Reggae/Latin": 8,
    "Experimental/Sound Art": 9, "Spoken/Soundtrack/Misc": 10}

    key_genre = st.selectbox("Select Genre:", list(genre_map.keys()))
    chosen_genre = genre_map[key_genre]
    length = st.slider("Select desired playlist length (songs):", 5, 30, 15)

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
    
    import pandas as pd
    from ast import literal_eval
    from random import choice
    
    gmi = pd.read_csv("data/genre_with_main_identity.csv")
    s_genres = gmi[["genre_id", "main_category_id"]]

    t = pd.read_csv("data/tracks_small.csv")
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
        
        rating = cols[4].radio(" ", ["👍", "👎"], horizontal=True, key=f"song_{idx}") 
        st.session_state.ratings[row["track_id"]] = rating

    if st.button("Generate Final Playlist"):
        st.session_state.evaluation_done = True
        st.session_state.step = 4
        st.success("Evaluation submitted! Proceed to Final Playlist.")


# -------------------------
# STEP 4 — Final Playlist
# -------------------------
if st.session_state.step >= 4 and st.session_state.evaluation_done:
    st.header("Step 4 – Your final recommended playlist")
    st.write("Generated based on your preferences and evaluations (mock results):")

    final_playlist = [
        {"Title": "Electric Feel", "Artist": "MGMT", "Score": random.randint(80, 99)},
        {"Title": "Peach", "Artist": "Kevin Abstract", "Score": random.randint(80, 99)},
        {"Title": "Golden Hour", "Artist": "JVKE", "Score": random.randint(80, 99)},
        {"Title": "Midnight City", "Artist": "M83", "Score": random.randint(80, 99)},
        {"Title": "Heat Waves", "Artist": "Glass Animals", "Score": random.randint(80, 99)},
    ]

    df_final = pd.DataFrame(final_playlist)
    st.dataframe(df_final, use_container_width=True)

    st.markdown("**Summary:**")
    st.write(f"- Total songs: {len(df_final)}")
    st.write(f"- Average recommendation score: {df_final['Score'].mean():.1f}%")

    if st.button("Start Over"):
        st.session_state.step = 1
        st.session_state.ratings = {}
        st.session_state.playlist_imported = False
        st.session_state.criteria_confirmed = False
        st.session_state.evaluation_done = False
        st.experimental_rerun()

    st.button("Save Playlist to Spotify (coming soon)")

