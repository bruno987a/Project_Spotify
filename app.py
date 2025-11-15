import streamlit as st
import pandas as pd
import random

# -------------------------
# 
# -------------------------
songs_data = [
    {"Title": "LDE", "Artist": "Taylor Swift", "Genre": "Pop", "Duration": "3:55"},
    {"Title": "Do I Wanna Know?", "Artist": "Arctic Monkeys", "Genre": "Indie Rock", "Duration": "4:33"},
    {"Title": "Bad Guy", "Artist": "Billie Eilish", "Genre": "Pop", "Duration": "3:14"},
    {"Title": "Blinding Lights", "Artist": "The Weeknd", "Genre": "Synthpop", "Duration": "3:20"},
    {"Title": "Levitating", "Artist": "Dua Lipa", "Genre": "Pop", "Duration": "3:23"},
]

candidate_songs = [
    {"Title": "Sunflower", "Artist": "Post Malone", "Genre": "Pop", "Mood": "Happy"},
    {"Title": "Lose Yourself", "Artist": "Eminem", "Genre": "Rap", "Mood": "Energetic"},
    {"Title": "Chill Bill", "Artist": "Rob $tone", "Genre": "Hip Hop", "Mood": "Chill"},
    {"Title": "Ocean Eyes", "Artist": "Billie Eilish", "Genre": "Pop", "Mood": "Calm"},
    {"Title": "Smells Like Teen Spirit", "Artist": "Nirvana", "Genre": "Rock", "Mood": "Energetic"},
]

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

    mood = st.selectbox("Select mood:",
    ["None", "Calm", "Energetic", "Happy", "Chill", "Sad"],
    index=0,  # default selection is "None"
    format_func=lambda x: f"*{x}*" if x=="None" else x,
    key="mood")
    length = st.slider("Select desired playlist length (songs):", 5, 30, 15, key="length")

    if st.button("Confirm and Continue"):
        st.session_state.criteria_confirmed = True
        st.session_state.step = 3
        st.success("Preferences saved. Proceed to Quick Evaluation.")


# -------------------------
# STEP 3 — Quick Evaluation
# -------------------------
if st.session_state.step >= 3 and st.session_state.criteria_confirmed:
    st.header("Step 3 – Quick song evaluation")
    st.write("Please like or dislike the following candidate songs:")

    # Display songs with inline rating buttons
    for idx, song in enumerate(candidate_songs):
        cols = st.columns([3, 3, 2, 2, 2])
        cols[0].write(song["Title"])
        cols[1].write(song["Artist"])
        cols[2].write(song["Genre"])
        cols[3].write(song["Mood"])
        rating = cols[4].radio(" ", ["👍", "👎"], horizontal=True, key=f"song_{idx}")
        st.session_state.ratings[song["Title"]] = rating

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
