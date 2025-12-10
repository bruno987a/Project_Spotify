# ----------------------------
# Set up all important imports
# ----------------------------

from pathlib import Path
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  # Machine Learning algorithm
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
from random import choice

candidate_songs = []

# ---------------------------------------------
# Set up connection to database for the project
# ---------------------------------------------

def get_conn():                                                     # Define get function to connect with sqlite3
    return sqlite3.connect("data/app.db")

DB = get_conn()                                                     # assign the database to the variable DB

st.set_page_config(
    page_title="Smart Playlist Generator",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Global Styles 
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
    /* Make sidebar wider */
    section[data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
    }
    /* Force selectbox (dropdown) background to white */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: #ffffff !important;
        border-radius: 0.6rem !important;
        border: 1px solid #d1d5db !important;
        box-shadow: none !important;
    }
    
    /
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


st.markdown(
    '<div class="main-title">Smart Playlist Generator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">'
    "Create group playlists that balance everyone’s taste."
    "</div>",
    unsafe_allow_html=True,
)


# --------- Setup (Step 0) ----------
if st.session_state.step >= 1:
    with st.container():
        # Invisible marker so the global CSS applies the card style
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Setup")
        st.caption(
            "Add everyone who will rate songs. We’ll combine all tastes into one smart playlist."
        )

        # BEFORE "Confirm group" is clicked → show editable inputs
        if st.session_state.step == 1:
            col1, col2 = st.columns([1, 2])

            with col1:
                num = st.number_input(
                    "Number of raters",
                    min_value=1,
                    max_value=10,
                    value=int(st.session_state.num_raters),
                    step=1,
                    key="num_raters_input",
                )

            names = []
            with col2:
                for i in range(int(num)):
                    default_name = (
                        st.session_state.rater_names[i]
                        if i < len(st.session_state.rater_names)
                        else f"User {i+1}"
                    )
                    names.append(
                        st.text_input(
                            f"Rater {i+1} name",
                            value=default_name,
                            key=f"rater_name_{i}",
                        )
                    )

            if st.button("✅ Confirm group & continue", use_container_width=True):
                clean_names = [(n.strip() or f"User {i+1}") for i, n in enumerate(names)]
                st.session_state.num_raters = int(num)
                st.session_state.rater_names = clean_names

                # initialize ratings dict per person
                st.session_state.ratings = {name: {} for name in clean_names}

                st.session_state.active_rater_idx = 0
                st.session_state.step = 2  # go to criteria step

                st.rerun()

        # AFTER confirm group → show summary
        else:
            total = st.session_state.num_raters
            names_display = ", ".join(st.session_state.rater_names)
            st.info(f"**Total raters:** {total} – {names_display}")



# -------------------------
# STEP 1 — Playlist generation criteria 
# -------------------------
if st.session_state.step >= 2:
    with st.container():
        # Required to trigger card CSS
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Playlist generation criteria")
        st.caption("Choose how focused the playlist should be and the kind of vibe you want.")

        # BEFORE confirming → show full criteria form
        if st.session_state.step == 2:

            col1, = st.columns(1)

            with col1:
                genre_map = {
                    "Rock/Metal/Punk": 1, "Pop/Synth": 2, "Electronic/IDM": 3,
                    "Hip-Hop/RnB": 4, "Jazz/Blues": 5, "Classical": 6,
                    "Folk/Country/Americana": 7, "World/Reggae/Latin": 8,
                    "Experimental/Sound Art": 9, "Spoken/Soundtrack/Misc": 10,
                    "Funk": 11
                }

                genre_raw = st.selectbox(
                    "Preferred genre",
                    list(genre_map.keys()),
                    index=None,                     
                    placeholder="Choose an option", 
                    key="genre_raw",
                )

            # Playlist length – full width
            st.session_state.n_desired_songs = st.slider(
                "Playlist length (number of songs)",
                5, 30, 15,
            )

            # Confirm button
            if st.button("✅ Confirm criteria & start rating", use_container_width=True):
                if genre_raw is None:
                    # "Popup" style warning 
                    st.markdown(
                        """
                        <div style="
                            padding: 0.8rem 1rem;
                            background-color: #fee2e2;
                            color: #b91c1c;
                            border: 1px solid #b91c1c;
                            border-radius: 0.6rem;
                            font-weight: 500;
                            margin-top: 0.5rem;
                            text-align: center;">
                            Please choose a preferred genre before continuing.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state.chosen_genre = genre_map[genre_raw]

                    st.session_state.criteria_confirmed = True
                    st.session_state.step = 3
                    st.session_state.evaluation_done = False
                    st.session_state.active_rater_idx = 0
                    if "candidate_songs" in st.session_state:
                        del st.session_state.candidate_songs

                    st.rerun()

        # AFTER confirming → summary card
        else:
            reverse_genre_map = {
                1: "Rock/Metal/Punk", 2: "Pop/Synth", 3: "Electronic/IDM",                
                4: "Hip-Hop/RnB", 5: "Jazz/Blues", 6: "Classical",
                7: "Folk/Country/Americana", 8: "World/Reggae/Latin",
                9: "Experimental/Sound Art", 10: "Spoken/Soundtrack/Misc",
                11: "Funk"
            }

            chosen_genre_id = st.session_state.get("chosen_genre")
            chosen_genre_name = reverse_genre_map.get(chosen_genre_id, "Unknown")

            st.info(
                f"""
**Criteria selected:**  
• Genre: **{chosen_genre_name}**  
• Desired playlist length: **{st.session_state.n_desired_songs} songs**
"""
            )




# -------------------------
# STEP 2 — Quick Evaluation 
# -------------------------
if st.session_state.step >= 3 and st.session_state.criteria_confirmed:
    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Quick song evaluation")

        if st.session_state.get("new_song_batch", False):                                           # Display this message if new song set had to be generated
            st.info(
            "A new set of song suggestions was generated because the group was unsatisfied with the previous ones."
            )
            st.session_state.new_song_batch = False                                                 # Only show it for the first user

        if st.session_state.num_raters > 1:
            st.caption(
                "Everyone rates a handful of songs. We’ll learn what the whole group likes and dislikes."
            )
        else:
            st.caption(
                "Please rate these songs. We'll learn what you like and dislike. You'll find the songs on the freemusicarchive.org webpage"
            )

        rater_names = st.session_state.rater_names
        idx_rater = st.session_state.active_rater_idx
        current_user = rater_names[idx_rater]

        st.write(f"**Rater {idx_rater + 1} / {len(rater_names)}:** {current_user}")
    

        # make sure this user's dict exists
        st.session_state.ratings.setdefault(current_user, {})
        user_ratings = st.session_state.ratings[current_user]

        # ===== Data loading for candidate songs =====

        gmi = pd.read_sql_query("SELECT * FROM genre_with_main_identity", DB)                                     #reading in the list with all subgenres linked with the main genres
        s_genres = gmi[["genre_id", "main_category_id"]]                                                          #filtering out the needed genre column
    
        t = pd.read_sql_query("SELECT * FROM tracks_small", DB)                                                   #importing the table with the tracks
        s_t = pd.DataFrame({                                                                                      #clean out the table whilst implementing it as a dataframe 
            "track_id": t["track_id"],
            "genres_all": t["genres_all"].fillna("[]").apply(literal_eval),                                       #we're safely changing the Genre numbers from type string to int format, empty ones would be transfered to []
            "title": t["title"],
            "artist": t["artist"]
        })
    
        def rand_track_genre(main_cat_id, n):                                                                     #implementing the function giving out random songs, with input of number of songs to rate (n) and the chosen main genre (main_cat_id) 
            genre_ids = list(set(s_genres.loc[s_genres["main_category_id"] == main_cat_id, "genre_id"]))          #constructing a list with all sub genres matching the chosen genre
            rand_gen_l = [choice(genre_ids) for _ in range(n)]                                                    #creating a list with n randomly chosen sub genres out the just created list
    
            p_to_rate = []
            for g_id in rand_gen_l:                                                                               #for every randomly chosen sub genre we choose one song that has this sub genre in the following lines
                poss_songs = s_t[s_t["genres_all"].apply(lambda ids: g_id in ids)]                                #we create a list of songs with the current sub genre g_id
                p_to_rate.append(poss_songs.sample(1))                                                            #one of the songs gets randomly chosen from this list and appended to the list of songs that will be displayed for rating
            return pd.concat(p_to_rate, ignore_index=True)                                                        #returning the created randomized selection of songs

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
                if st.button("🎉 Generate final playlist", type="primary", use_container_width=True):

                    # ==== Implement a check if at least half the group is happy with the suggested songs ====
                    rater_names = st.session_state.rater_names                                                # List of all raters
                    num_raters = len(rater_names)                                                             # Number of users for average

                    unhappy_count = 0                                                                         # Initialize a count for unhappy users
                    track_ids = songs_df["track_id"].tolist()                                                 # List of all track IDs for max ratings
                    
                    for name in rater_names:                                                                  # Iterate over names of users
                        rating_dict = st.session_state.ratings.get(name, {})                                  # Retrieve this user's rating
                        
                        max_r = max(                                                                          # retrieve the maximum number of points given by each user
                            rating_dict.get(t_id, 1) for t_id in track_ids
                        )
                        
                        if max_r < 3:                                                                        # sum all 'unhappy' users into the following counter, if maximum rating is below 3
                            unhappy_count += 1

                    # if more than half of the users are unhappy with the randomized song selection, rerun the whole process:
                    if unhappy_count > num_raters / 2:
                        
                        st.session_state.ratings = {name: {} for name in rater_names}                       # reset all the ratings before repeating process

                        if "candidate_songs" in st.session_state:                                           # discard previously selected songs for new selection process
                            del st.session_state.candidate_songs

                        st.session_state.active_rater_idx = 0                                              # Start over with Rater (user) 1
                        st.session_state.evaluation_done = False                                           # mark evaluation as unfinished
                        st.session_state.step = 3                                                          # return to quick evaluation Step 3

                        st.session_state.new_song_batch = True                                             # Show info message in repeated selection round

                        st.rerun()

                    else:

                        # ------------------------------
                        # START MACHINE LEARNING PART 
                        # ------------------------------
                        features = pd.read_sql("SELECT * FROM features", DB, index_col="track_id")

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
                        
                        def build_user_profile(ratings_list, rated_ids, features_df):               # define function to create weighted vectors for each user's preferences
                            ratings = np.asarray(ratings_list, dtype=float)                         # Convert list of ratings to Numpy array
                            vecs = features_df.loc[rated_ids].values                                # Get the feature rows of the rated songs
                            return np.average(vecs, axis=0, weights=ratings)                        # Compute the weighted average to receive one user's vector
                        
                        def weight_adjustment(points: int) -> float:                                # define function to weaken/ enhance song features of less/ more liked songs
                            return (points / 3.0) ** 2                                              # Significantly weakens songs with rating of below 3, keeps 3 neutral, and enhances more liked ones
                        
                        user_profiles = []                                                          # set up empty list and append each user's vector
                        for _, rating_dict in st.session_state.ratings.items():                     # iterate over each users ratings
                            if not rating_dict:                                                     # Skip user in case rating wasn't completed
                                continue

                            rated_ids = [t_id for t_id in rating_dict.keys()                        # all track IDs of the rated songs
                                         if t_id in features_14_scaled.index]                       # ...if they exist in the feature table

                            ratings_list = [weight_adjustment(rating_dict[t_id])                    # adjusted list of ratings with non-linear weaking / enhancement
                                            for t_id in rated_ids]
                            user_profiles.append(                                                   # Calculate vector to represent one user's taste
                                build_user_profile(ratings_list, rated_ids, features_14_scaled)
                            )

                        group_vector = np.mean(user_profiles, axis=0)                               # Derive group vector from average of all individual vectors

                        X = features_14_scaled.values                                               # convert features to Numpy values
                        track_ids = features_14_scaled.index.to_numpy()                             # save their corresponding IDs

                        knn_model = NearestNeighbors(metric="cosine", n_neighbors=200)              # kNN model with cosine similarity
                        knn_model.fit(X)                                                            # load the song features into the model

                        def recommend(group_vec, n_songs):                                          # define function for song recommendation
                            _, nn_idx = knn_model.kneighbors(group_vec.reshape(1, -1), n_neighbors=n_songs)
                            return track_ids[nn_idx[0]]

                        recommended_ids = recommend(group_vector, st.session_state.n_desired_songs).tolist()    # Call function and assign to easy variable for final display

                        # store for step 4
                        st.session_state.recommended_ids = recommended_ids
                        st.session_state.evaluation_done = True
                        st.session_state.step = 4

                        # 🔹 tell next run to show success message on the final page
                        st.session_state.final_success_message = True

                        # 🔹 force rerun so sidebar + final playlist update immediately
                        st.rerun() 

# -------------------------
# STEP 3 — Final Playlist 
# -------------------------
if st.session_state.step >= 4 and st.session_state.evaluation_done:
    with st.container():
        st.markdown('<div class="step-card"></div>', unsafe_allow_html=True)

        st.markdown("### Final recommended playlist")
        if st.session_state.final_success_message:
            if st.session_state.num_raters > 1:
                msg = "✅ Playlist generated based on the whole group’s preferences!"
            else:
                msg = "✅ Playlist generated based on your preferences!"
            st.success(msg)
            st.session_state.final_success_message = False

        t = pd.read_sql_query("SELECT * FROM tracks_small", DB)
        s_t = pd.DataFrame({
            "track_id": t["track_id"],
            "title": t["title"],
            "artist": t["artist"]
        })

        df_final = s_t[s_t["track_id"].isin(st.session_state.recommended_ids)][["title", "artist"]]
        df_final_display = df_final.reset_index(drop=True)
        df_final_display.index = df_final_display.index + 1

        st.dataframe(
            df_final_display,
            use_container_width=True
        )
    # ---------------------------------------------
    # Simple visualization: distribution of ratings
    # ---------------------------------------------
    
    if "candidate_songs" in st.session_state and st.session_state.ratings:
        songs_df = st.session_state.candidate_songs.reset_index(drop=True)

        # Numeric x positions: 1, 2, 3, ...
        x_positions = np.arange(1, len(songs_df) + 1)
        song_labels = [f"Song {i+1}" for i in range(len(songs_df))]

        with st.expander("📊 Show rating distribution per song and user"):
            st.write(
                "This line chart shows how each user rated each of the quick-evaluation songs."
            )

            fig, ax = plt.subplots(figsize=(6, 3.5))

            for username in st.session_state.rater_names:
                rating_dict = st.session_state.ratings.get(username, {})
                user_values = []

                # Collect ratings in the same order as songs_df
                for _, row in songs_df.iterrows():
                    track_id = row["track_id"]
                    user_values.append(rating_dict.get(track_id, np.nan))

                user_values = np.array(user_values, dtype=float)

                # If user has no valid ratings, skip them
                if np.all(np.isnan(user_values)):
                    continue

                ax.plot(
                    x_positions,
                    user_values,
                    marker="o",
                    linestyle="-",
                    label=username,
                )

            # Axis formatting
            ax.set_xlim(1, len(songs_df))
            ax.set_ylim(0, 5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(song_labels)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Song in quick evaluation")
            ax.set_ylabel("Rating")
            ax.set_title("Ratings per song for each rater")

            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(title="Rater:", bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()

            st.pyplot(fig)
            # ---------- 2) BAR CHART: average rating per song ----------
            # Compute average rating for each song across all users
            avg_values = []
            avg_labels = []

            for i, row in songs_df.iterrows():
                track_id = row["track_id"]
                song_ratings = []

                for username in st.session_state.rater_names:
                    rating_dict = st.session_state.ratings.get(username, {})
                    val = rating_dict.get(track_id, np.nan)
                    if not np.isnan(val):
                        song_ratings.append(val)

                if song_ratings:
                    avg = float(np.mean(song_ratings))
                    avg_values.append(avg)
                    avg_labels.append(f"{i+1}")

            if avg_values:
                y_pos = np.arange(len(avg_labels))

                # ✨ Define 5 colors (you can change these hex codes as you like)
                bar_colors = [
                    "#6366F1",  # indigo
                    "#22C55E",  # green
                    "#F97316",  # orange
                    "#EC4899",  # pink
                    "#14B8A6",  # teal
                ]

                fig2, ax2 = plt.subplots(figsize=(6, 3.5))
                bars = ax2.barh(
                    y_pos,
                    avg_values,
                    color=[bar_colors[i % len(bar_colors)] for i in range(len(avg_labels))]
                )

                ax2.set_xlim(0, 5)  # ratings from 1 to 5
                ax2.set_xticks([0, 1, 2, 3, 4, 5])
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(avg_labels)
                ax2.set_xlabel("Average rating")
                ax2.set_ylabel("Song")
                ax2.set_title("Average rating per song")
                ax2.grid(True, axis="x", linestyle="--", alpha=0.3)

                # Put the numeric value at the end of each bar
                for i, v in enumerate(avg_values):
                    ax2.text(
                        v + 0.05,       # a bit to the right of bar end
                        i,
                        f"{v:.2f}",     # e.g. 3.67
                        va="center",
                    )

                fig2.tight_layout()
                st.pyplot(fig2)


    else:
        st.info("No quick evaluation songs available to show a rating chart yet.")




st.markdown(
    '<div class="footer">© Smart Playlist</div>',
    unsafe_allow_html=True
)
