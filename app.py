import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.neighbors import NearestNeighbors  # Machine Learning algorithm @Lorenz
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
from random import choice

candidate_songs = []

# Set up pathways to data folder
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# Set up SQLite connection to database as DB
def get_conn():
    return sqlite3.connect(DATA_DIR / "app.db")


DB = get_conn()

st.set_page_config(
    page_title="Smart Playlist Generator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Global Styles (design de Code 1)
# -------------------------
st.markdown(
    """
    <style>
    /* Make global text bigger */
    html, body, .stApp {
        font-size: 18px;  /* base size up from default */
    }

    /* Overall background – white */
    .stApp {
        background: #ffffff;
    }

    .main-title {
        font-size: 3rem;          /* bigger main title */
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .main-subtitle {
        font-size: 1.1rem;        /* bigger subtitle */
        color: #666;
        margin-bottom: 1.2rem;
    }

    /* Cards for each step – tighter padding so content starts closer to the edge */
    .block-container div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]:has(.step-card) {
        background: radial-gradient(circle at top left, #fdfbfb 0, #ebedee 40%, #f7f7f7 100%);
        border-radius: 1.1rem;
        padding: 0.8rem 1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin-bottom: 1rem;
    }

    /* Invisible marker for step cards */
    .step-card {
        height: 0;
        margin: 0;
        padding: 0;
    }

    /* WHITE input backgrounds (number, text, select) */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: #ffffff !important;
        border-radius: 0.6rem !important;
        border: 1px solid #d1d5db !important;
        font-size: 1rem !important;  /* make input text bigger */
    }

    /* WHITE + / - buttons on number input */
    div[data-testid="stNumberInput"] button {
        background-color: #ffffff !important;
        border-radius: 0.6rem !important;
        border: 1px solid #d1d5db !important;
        font-size: 1rem !important;
    }

    /* Make most labels and normal text a bit larger */
    label, .stMarkdown p, .stMarkdown li, .stCheckbox, .stRadio, .stSlider label {
        font-size: 1rem !important;
    }

    /* Sidebar steps */
    .step-label {
        padding: 0.35rem 0.5rem;
        border-radius: 0.8rem;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .step-done {
        background: rgba(34, 197, 94, 0.12);
        color: #166534;
    }
    .step-current {
        background: rgba(59, 130, 246, 0.12);
        color: #1d4ed8;
    }
    .step-todo {
        background: rgba(148, 163, 184, 0.12);
        color: #475569;
    }

    /* Song list rows */
    .song-row {
        background: #fafafa;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .song-title {
        font-weight: 600;
        font-size: 1rem;
        color: #111827;
    }
    .song-artist {
        font-size: 0.9rem;
        color: #6b7280;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 20px 0 5px 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Session state
# -------------------------
if "step" not in st.session_state:
    st.session_state.step = 1   # start at group setup

if "num_raters" not in st.session_state:
    st.session_state.num_raters = 1

if "rater_names" not in st.session_state:
    st.session_state.rater_names = ["User 1"]

if "active_rater_idx" not in st.session_state:
    st.session_state.active_rater_idx = 0

if "ratings" not in st.session_state:
    st.session_state.ratings = {}

if "criteria_confirmed" not in st.session_state:
    st.session_state.criteria_confirmed = False

if "evaluation_done" not in st.session_state:
    st.session_state.evaluation_done = False

if "final_success_message" not in st.session_state:
    st.session_state.final_success_message = False

# store preferences
if "chosen_genre" not in st.session_state:
    st.session_state.chosen_genre = None

if "n_desired_songs" not in st.session_state:
    st.session_state.n_desired_songs = 15

# -------------------------
# Sidebar: progress indicator
# -------------------------
def render_sidebar():
    st.sidebar.title("Progress")

    step0_done = st.session_state.step > 1
    step1_done = st.session_state.criteria_confirmed
    step2_done = st.session_state.evaluation_done
    step3_done = st.session_state.step >= 4

    steps = [
        ("Step 0 – Setup", step0_done),
        ("Step 1 – Criteria", step1_done),
        ("Step 2 – Quick evaluation", step2_done),
        ("Step 3 – Final playlist", step3_done),
    ]

    current_step = st.session_state.step

    for idx, (label, done) in enumerate(steps, start=1):
        if done:
            css = "step-label step-done"
            icon = "✅"
        elif idx == current_step:
            css = "step-label step-current"
            icon = "▶️"
        else:
            css = "step-label step-todo"
            icon = "▫️"

        st.sidebar.markdown(
            f"<div class='{css}'>{icon} {label}</div>", unsafe_allow_html=True
        )

render_sidebar()

# -------------------------
# Titre principal (design Code 1)
# -------------------------
st.markdown(
    '<div class="main-title">Smart Playlist Generator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    "Create group playlists that balance everyone’s taste.."
    "</div>",
    unsafe_allow_html=True,
)

# -------------------------
# ---------Group Setup (Step 0) dans une card
# -------------------------
if st.session_state.step >= 1:
    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Setup")
        st.caption("Add everyone who will rate songs. We’ll combine all tastes into one smart playlist.")

        # BEFORE "Confirm group" is clicked → show editable inputs
        if st.session_state.step == 1:
            num = st.number_input(
                "Number of raters",
                min_value=1,
                max_value=10,
                value=int(st.session_state.num_raters),
                step=1,
                key="num_raters_input"
            )

            # name inputs
            names = []
            for i in range(int(num)):
                default_name = (
                    st.session_state.rater_names[i]
                    if i < len(st.session_state.rater_names)
                    else f"User {i+1}"
                )
                names.append(
                    st.text_input(f"Rater {i+1} name", value=default_name, key=f"rater_name_{i}")
                )

            if st.button("✅ Confirm group & continue", use_container_width=True):
                clean_names = [(n.strip() or f"User {i+1}") for i, n in enumerate(names)]
                st.session_state.num_raters = int(num)
                st.session_state.rater_names = clean_names

                # initialize ratings dict per person
                st.session_state.ratings = {name: {} for name in clean_names}

                st.session_state.active_rater_idx = 0
                st.session_state.step = 2  # GO TO STEP 1

                st.rerun()

        # AFTER confirm group → show summary 
        else:
            total = st.session_state.num_raters
            names_display = ", ".join(st.session_state.rater_names)
            st.info(f"**Total raters:** {total} – {names_display}")


# -------------------------
# STEP 1 — Generation Criteria (en card)
# -------------------------
if st.session_state.step >= 2:
    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Playlist generation criteria")
        st.caption("Choose how focused the playlist should be and the kind of vibe you want.")

        # BEFORE confirming → show full criteria form
        if st.session_state.step == 2:
            similarity = st.selectbox(
                "Similarity level",
                ["None", "Genre", "Artist", "Mixed"],
                index=0,  # default selection is "None"
                format_func=lambda x: f"*{x}*" if x == "None" else x,
                key="similarity"
            )

            # Song selection for rating 
            genre_map = {
                "Rock/Metal/Punk": 1, "Pop/Synth": 2, "Electronic/IDM": 3,
                "Hip-Hop/RnB": 4, "Jazz/Blues": 5, "Classical": 6,
                "Folk/Country/Americana": 7, "World/Reggae/Latin": 8,
                "Experimental/Sound Art": 9, "Spoken/Soundtrack/Misc": 10,
                "Funk": 11
            }

            key_genre = st.selectbox("Preferred genre", list(genre_map.keys()))
            st.session_state.chosen_genre = genre_map[key_genre]
            st.session_state.n_desired_songs = st.slider(
                "Playlist length (number of songs)",
                5, 30, 15
            )

            # button for continuing the workflow and start the rating process    
            if st.button("✅ Confirm criteria & start rating", use_container_width=True):
                st.session_state.criteria_confirmed = True
                st.session_state.step = 3
                st.session_state.evaluation_done = False
                st.session_state.active_rater_idx = 0
                if "candidate_songs" in st.session_state:
                    del st.session_state.candidate_songs

                st.rerun()

        # AFTER confirming → show criteria summary
        else:
            reverse_genre_map = {
                1: "Rock/Metal/Punk", 2: "Pop/Synth", 3: "Electronic/IDM",
                4: "Hip-Hop/RnB", 5: "Jazz/Blues", 6: "Classical",
                7: "Folk/Country/Americana", 8: "World/Reggae/Latin",
                9: "Experimental/Sound Art", 10: "Spoken/Soundtrack/Misc",
                11: "Funk"
            }

            # SAFE access with defaults
            similarity_value = st.session_state.get("similarity", "None")
            chosen_genre_id = st.session_state.get("chosen_genre")
            chosen_genre_name = reverse_genre_map.get(chosen_genre_id, "Unknown")

            st.info(
                f"""
🎛️ **Criteria selected**  
• Similarity level: **{similarity_value}**  
• Genre: **{chosen_genre_name}**  
• Desired playlist length: **{st.session_state.n_desired_songs} songs**
"""
            )


# -------------------------
# STEP 2 — Quick Evaluation (en card, mais ML identique à Code 2)
# -------------------------
if st.session_state.step >= 3 and st.session_state.criteria_confirmed:
    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Quick song evaluation")

        rater_names = st.session_state.rater_names
        idx_rater = st.session_state.active_rater_idx
        current_user = rater_names[idx_rater]

        st.write(f"**Rater {idx_rater + 1} / {len(rater_names)}:** {current_user}")
        st.caption("Please rate the following songs. 1 = dislike · 5 = love")

        # make sure this user's dict exists
        st.session_state.ratings.setdefault(current_user, {})
        user_ratings = st.session_state.ratings[current_user]

        # ===== Data loading for candidate songs =====
        gmi = pd.read_sql_query("SELECT * FROM genre_with_main_identity", DB)  # subgenres + main genres
        s_genres = gmi[["genre_id", "main_category_id"]]

        t = pd.read_sql_query("SELECT * FROM tracks_small", DB)
        s_t = pd.DataFrame({
            "track_id": t["track_id"],
            "genres_all": t["genres_all"].fillna("[]").apply(literal_eval),
            "title": t["title"],
            "artist": t["artist"]
        })

        def rand_track_genre(main_cat_id, n):
            genre_ids = list(set(s_genres.loc[s_genres["main_category_id"] == main_cat_id, "genre_id"]))
            rand_gen_l = [choice(genre_ids) for _ in range(n)]

            p_to_rate = []
            for g_id in rand_gen_l:
                poss_songs = s_t[s_t["genres_all"].apply(lambda ids: g_id in ids)]
                if len(poss_songs) == 0:
                    continue
                p_to_rate.append(poss_songs.sample(1))
            if not p_to_rate:
                return s_t.sample(n=min(n, len(s_t)))
            return pd.concat(p_to_rate, ignore_index=True)

        # Generate candidate songs ONCE for the whole group
        if "candidate_songs" not in st.session_state:
            st.session_state.candidate_songs = rand_track_genre(st.session_state.chosen_genre, 5)

        songs_df = st.session_state.candidate_songs

        # Header row (Songs / Rating)
        header_song_col, header_rating_col = st.columns([3, 2])
        with header_song_col:
            st.markdown(
                "<div style='background-color:#e5e7eb; padding:0.5rem; "
                "border-radius:0.75rem 0 0 0.75rem; font-weight:600;'>Songs</div>",
                unsafe_allow_html=True,
            )

        with header_rating_col:
            st.markdown(
                "<div style='background-color:#e5e7eb; padding:0.5rem; "
                "border-radius:0 0.75rem 0.75rem 0; font-weight:600; display:flex; "
                "justify-content:space-between; align-items:center;'>"
                "<span>Rating</span>"
                "<span style='font-weight:400; font-size:0.8rem;'>1 = dislike · 5 = love</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Data rows
        for i, row in songs_df.iterrows():
            song_col, rating_col = st.columns([3, 2])

            # SONG CELL
            with song_col:
                st.markdown(
                    f"""
                    <div class="song-row">
                        <div class="song-title">{row['title']}</div>
                        <div class="song-artist">{row['artist']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # RATING CELL (NO LABEL)
            with rating_col:
                rating = st.slider(
                    label="",
                    min_value=1,
                    max_value=5,
                    value=int(user_ratings.get(row["track_id"], 3)),
                    key=f"rating_{current_user}_{i}",
                    step=1,
                )
                user_ratings[row["track_id"]] = rating

        # Buttons to go to next person
        if idx_rater < len(rater_names) - 1:
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                if st.button("➡️ Save ratings & next person", use_container_width=True):
                    st.session_state.active_rater_idx += 1
                    st.rerun()
        else:
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                # allow generation for last rater
                if st.button("🎉 Generate final playlist", use_container_width=True):

                    # ------------------------------
                    # START MACHINE LEARNING PART (comme ton Code 2)
                    # ------------------------------
                    features = pd.read_csv(DATA_DIR / "reduced_features.csv", index_col=0)

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

                    def build_user_profile(ratings_list, rated_ids, features_df):
                        ratings = np.asarray(ratings_list, dtype=float)
                        vecs = features_df.loc[rated_ids].values
                        return np.average(vecs, axis=0, weights=ratings)
                    
                    def weight_adjustment(points: int) -> float:
                        return (points / 3.0) ** 2

                    user_profiles = []
                    for username, rating_dict in st.session_state.ratings.items():
                        if not rating_dict:
                            continue

                        rated_ids = [tid for tid in rating_dict.keys() if tid in features_14_scaled.index]
                        if not rated_ids:
                            continue

                        ratings_list = [weight_adjustment(rating_dict[tid]) for tid in rated_ids]
                        user_profiles.append(build_user_profile(ratings_list, rated_ids, features_14_scaled))

                    if len(user_profiles) == 0:
                        st.error("There are no usable ratings - no recommendation possible.")
                        st.stop()

                    group_vector = np.mean(user_profiles, axis=0)

                    X = features_14_scaled.values
                    track_ids = features_14_scaled.index.to_numpy()

                    knn_model = NearestNeighbors(metric="cosine", n_neighbors=200)
                    knn_model.fit(X)

                    def recommend(group_vec, n_songs):
                        _, nn_idx = knn_model.kneighbors(group_vec.reshape(1, -1), n_neighbors=n_songs)
                        return track_ids[nn_idx[0]]

                    recommended_ids = recommend(group_vector, st.session_state.n_desired_songs).tolist()

                    # store for step 4
                    st.session_state.recommended_ids = recommended_ids
                    st.session_state.evaluation_done = True
                    st.session_state.step = 4

                    # 🔹 tell next run to show success message on the final page
                    st.session_state.final_success_message = True

                    # 🔹 force rerun so sidebar + final playlist update immediately
                    st.rerun() 

# -------------------------
# STEP 3 — Final Playlist (en card)
# -------------------------
if st.session_state.step >= 4 and st.session_state.evaluation_done:
    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Final recommended playlist")
        st.write("Generated based on your preferences and evaluations:")

        # s_t vient de plus haut (dans step 2), on le recharge au cas où
        t = pd.read_sql_query("SELECT * FROM tracks_small", DB)
        s_t = pd.DataFrame({
            "track_id": t["track_id"],
            "title": t["title"],
            "artist": t["artist"]
        })

        df_final = s_t[s_t["track_id"].isin(st.session_state.recommended_ids)][["title", "artist"]]
        st.dataframe(df_final.reset_index(drop=True), use_container_width=True)

        st.markdown("**Summary:**")
        st.write(f"- Total songs: {len(df_final)}")

        if st.button("🔁 Start over", use_container_width=True):
            # Completely clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]

            # Rerun to reinitialize everything
            st.rerun()

# Footer avec style Code 1
st.markdown(
    '<div class="footer">© 2025 Cookable</div>',
    unsafe_allow_html=True
)
