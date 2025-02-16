import os
import sys
import logging
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy.sql import text

# Add relevant folders to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
components_path = os.path.join(project_root, "components")
recommendation_path = os.path.join(project_root, "recommendation")

sys.path.insert(0, components_path)
sys.path.insert(0, recommendation_path)

from components.recommendations_ui import RecommendationUI
from components.utils import (
    get_movie_id_by_title,
    get_sqlalchemy_engine, 
    search_tmdb_actors, 
    fetch_model_metrics, 
    fetch_unique_genres, 
    fetch_unique_directors
)

# Import recommendation modules
from recommendation.narrative_recommender_streamlit import (
    generate_narrative_recommendations,
    generate_hybrid_recommendations,
    preprocess_user_input,
    similarity_analysis,
    metadata_evaluation
)
from recommendation.interaction_handler_streamlit import (
    quiz_based_recommendations,
    train_interactive_model,
    fetch_user_insights,
    log_recommendation
)

rec_ui = RecommendationUI()

def global_styling():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500&display=swap');

            /* Apply Outfit font globally */
            body, h1, h2, h3, h4, h5, h6, p, div {
                font-family: 'Outfit', sans-serif;
                font-weight: 500;
            }

            /* Dark Background and General Text Styling */
            body {
                background-color: #121212;
                color: #EDEDED;
            }

            /* Intro Section Styling */
            .intro-section {
                background-color: #1F1F1F;
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin-bottom: 20px;
            }

            .intro-section h2 {
                color: #FFAA00; /* Highlighted section title */
                font-weight: 600;
            }

            .intro-section p {
                color: #CCCCCC; /* Subtle text for descriptions */
                font-size: 16px;
                line-height: 1.6;
            }

            /* Feature Card Styling */
            .feature-card {
                background-color: #2D2D2D;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                color: #F1F1F1;
            }

            .feature-card h3 {
                font-size: 22px;
                color: #FFAA00; /* Title color for cards */
                margin-bottom: 10px;
            }

            .feature-card p {
                font-size: 15px;
                color: #BBBBBB; /* Lighter gray for descriptions */
                line-height: 1.5;
            }

            /* Quiz Section Styling */
            .quiz-section {
                background-color: #2D2D2D;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
            }

            .quiz-section h4 {
                color: #FFAA00;
                margin-bottom: 10px;
                font-size: 18px;
            }

            .quiz-section p {
                color: #CCCCCC;
                font-size: 14px;
                line-height: 1.5;
            }

            /* Custom Button Styling */
            .custom-button {
                background-color: #28a745; /* Green button */
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }

            .custom-button:hover {
                background-color: #218838; /* Darker green on hover */
            }

            .custom-button:focus {
                outline: none;
                box-shadow: 0 0 5px #28a745;
            }

            /* Detailed List Items */
            .detailed-item {
                color: #90CAF9; /* Light blue for easy scanning */
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 5px;
            }

            /* Section Headers */
            h3, h4 {
                color: #FFAA00;
                margin-bottom: 10px;
            }

            /* Metric Section Styling */
            .stMetricValue {
                color: #FFD700; /* Gold color for metrics */
                font-weight: bold;
            }

            /* Charts */
            .st-bar-chart svg {
                background-color: #1F1F1F; /* Match the dark theme */
            }

            /* Table Headers */
            .dataframe thead th {
                background-color: #333333; /* Dark table header */
                color: #FFAA00; /* Gold text for headers */
                font-size: 16px;
            }

            /* Table Body */
            .dataframe tbody tr {
                background-color: #2D2D2D;
                color: #EDEDED;
            }

            .dataframe tbody tr:hover {
                background-color: #444444; /* Hover effect for table rows */
            }
        </style>
    """, unsafe_allow_html=True)

# Welcome Menu
def welcome_menu():
    global_styling()
    st.title("🎥 Welcome to CineSphere!")

    # Introduction Section
    st.markdown("""
    <div class="intro-section">
        <h2>What is CineSphere?</h2>
        <p>CineSphere is your personal movie recommendation assistant, combining the power of AI with the magic of storytelling. Whether you're in the mood for thrilling car chases, heartwarming romances, or epic space adventures, CineSphere understands your preferences and delivers tailored movie recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.subheader("Here's Why You'll Love CineSphere:")
    features = [
        {
            "icon": "🌟",
            "title": "Tailored Recommendations",
            "description": "Find movies perfectly aligned with your mood and preferences."
        },
        {
            "icon": "📊",
            "title": "Insightful Analytics",
            "description": "Discover your most-loved genres, directors, and themes."
        },
        {
            "icon": "🎬",
            "title": "Scene-Based Discovery",
            "description": "Love a specific type of scene? CineSphere has you covered."
        },
        {
            "icon": "🎞️",
            "title": "Recommendation History",
            "description": "Track your past recommendations and feedback."
        },
        {
            "icon": "📝",
            "title": "Personalized Profiles",
            "description": "Customize your profile with genres, directors, and styles you adore."
        }
    ]

    # Render feature cards
    for feature in features:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{feature['icon']}</div>
                <div>
                    <div class="feature-title">{feature['title']}</div>
                    <div class="feature-description">{feature['description']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Call-to-Action Section
    st.markdown("""
    ---
    ## Ready to Personalize Your Experience?
    Take a moment to set up your preferences below. This helps us make the best recommendations for you!
    """)

    # Persistent State for Inputs
    if "welcome_preferences" not in st.session_state:
        st.session_state["welcome_preferences"] = {
            "favorite_movies": [],
            "preferred_genres": [],
            "preferred_styles": "",
            "preferred_directors": [],
            "preferred_actors": []
        }

    # Singleton Database Connection
    engine = get_sqlalchemy_engine()
    if not engine:
        st.error("Database connection failed. Please contact support.")
        return

    # Fetch data from the database only once
    try:
        with engine.connect() as conn:
            available_movies = [row[0] for row in conn.execute(text("SELECT title FROM movies")).fetchall()]
            available_genres = [
                row[0] for row in conn.execute(
                    text("SELECT DISTINCT UNNEST(STRING_TO_ARRAY(genre, ', ')) AS genre FROM movies")
                ).fetchall()
            ]
            available_directors = [row[0] for row in conn.execute(text("SELECT DISTINCT director FROM movies")).fetchall()]
    except Exception as e:
        st.error("Failed to fetch data. Please contact support.")
        logging.error(f"Database error: {e}")
        available_movies, available_genres, available_directors = [], [], []

    # Handle favorite movies
    st.subheader("Choose Movies You Like:")
    movies_selected = st.multiselect(
        "Select Movies",
        options=available_movies,
        default=st.session_state["welcome_preferences"]["favorite_movies"],
        key="favorite_movies"
    )
    if movies_selected != st.session_state["welcome_preferences"]["favorite_movies"]:
        st.session_state["welcome_preferences"]["favorite_movies"] = movies_selected

    # Handle preferred genres
    st.subheader("Select Preferred Genres:")
    genres_selected = st.multiselect(
        "Select Genres",
        options=available_genres,
        default=st.session_state["welcome_preferences"]["preferred_genres"],
        key="preferred_genres"
    )
    if genres_selected != st.session_state["welcome_preferences"]["preferred_genres"]:
        st.session_state["welcome_preferences"]["preferred_genres"] = genres_selected

    # Handle preferred styles
    st.subheader("Enter Preferred Styles:")
    preferred_styles = st.text_area(
        "Enter preferred styles (e.g., Dark, Thriller, Humorous):",
        value=st.session_state["welcome_preferences"]["preferred_styles"],
        key="preferred_styles"
    )
    st.session_state["welcome_preferences"]["preferred_styles"] = preferred_styles

    # Handle preferred directors
    st.subheader("Select Preferred Directors:")
    directors_selected = st.multiselect(
        "Select Directors",
        options=available_directors,
        default=st.session_state["welcome_preferences"]["preferred_directors"],
        key="preferred_directors"
    )
    if directors_selected != st.session_state["welcome_preferences"]["preferred_directors"]:
        st.session_state["welcome_preferences"]["preferred_directors"] = directors_selected

    # Collect preferred actors dynamically with persistence
    st.subheader("Search and Select Preferred Actors:")
    actor_query = st.text_input("Search for an actor by name:", key="actor_query")

    # Persist previously selected actors
    if "preferred_actors" not in st.session_state["welcome_preferences"]:
        st.session_state["welcome_preferences"]["preferred_actors"] = []

    # Search actors dynamically
    if actor_query.strip():
        try:
            actors_list = search_tmdb_actors(actor_query, engine)
            if actors_list:
                # Ensure default values are valid options
                valid_default_actors = [
                    actor for actor in st.session_state["welcome_preferences"]["preferred_actors"]
                    if actor in actors_list
                ]
                selected_actors = st.multiselect(
                    "Select Actors",
                    actors_list,
                    default=valid_default_actors,
                    key="preferred_actors"
                )
                # Merge new selections with previously selected actors
                st.session_state["welcome_preferences"]["preferred_actors"] = list(
                    set(valid_default_actors + selected_actors)
                )
            else:
                st.warning("No actors found matching your search. Try a different query.")
        except Exception as e:
            st.error("Error fetching actors. Please try again later.")
            logging.error(f"Error searching actors: {e}")
    else:
        selected_actors = st.session_state["welcome_preferences"]["preferred_actors"]

    # Display currently selected actors
    st.write("Currently Selected Actors:")
    st.write(st.session_state["welcome_preferences"]["preferred_actors"])

    # Save preferences
    if st.button("Save Preferences", key="save_preferences"):
        try:
            preferences = st.session_state["welcome_preferences"]
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE users
                        SET
                            favorite_movies = :favorite_movies,
                            preferred_genres = :preferred_genres,
                            preferred_styles = :preferred_styles,
                            preferred_directors = :preferred_directors,
                            preferred_actors = :preferred_actors
                        WHERE username = :username
                    """),
                    {
                        "favorite_movies": ",".join(preferences["favorite_movies"]),
                        "preferred_genres": ",".join(preferences["preferred_genres"]),
                        "preferred_styles": preferences["preferred_styles"],
                        "preferred_directors": ",".join(preferences["preferred_directors"]),
                        "preferred_actors": ",".join(preferences["preferred_actors"]),
                        "username": st.session_state["USERNAME"],
                    }
                )
            st.success("Preferences saved successfully!")
        except Exception as e:
            st.error("Failed to save preferences.")
            logging.error(f"Error saving preferences: {e}")

# Profile Section
def profile_section():
    global_styling()
    st.title("Profile Management")
    engine = get_sqlalchemy_engine()

    # Fetch user profile and preferences
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT name, email, favorite_movies, preferred_genres, preferred_styles, preferred_directors
                FROM users
                WHERE username = :username
            """)
            user_profile = conn.execute(query, {"username": st.session_state["USERNAME"]}).fetchone()

        if not user_profile:
            st.error("User profile not found. Please ensure your account exists.")
            return

        # Unpack user profile fields
        name = user_profile[0]
        email = user_profile[1]
        favorite_movies = user_profile[2].split(",") if user_profile[2] else []
        preferred_genres = user_profile[3].split(",") if user_profile[3] else []
        preferred_styles = user_profile[4] or ""
        preferred_directors = user_profile[5].split(",") if user_profile[5] else []
    except Exception as e:
        st.error("Failed to fetch user profile. Please try again later.")
        logging.error(f"Error fetching user profile: {e}")
        return

    # Display editable user information
    name = st.text_input("Full Name", value=name, key="profile_name")
    email = st.text_input("Email", value=email, key="profile_email")

    # Update Profile Information
    if st.button("Update Profile"):
        try:
            with engine.connect() as conn:
                conn.execute(
                    text("UPDATE users SET name = :name, email = :email WHERE username = :username"),
                    {"name": name, "email": email, "username": st.session_state["USERNAME"]}
                )
            st.success("Profile updated successfully!")
        except Exception as e:
            st.error("Failed to update profile.")
            logging.error(f"Error updating profile for {st.session_state['USERNAME']}: {e}")

    # Fetch `user_id`
    try:
        with engine.connect() as conn:
            user_id_query = text("SELECT user_id FROM users WHERE username = :username")
            user_id_result = conn.execute(user_id_query, {"username": st.session_state["USERNAME"]}).fetchone()

        if not user_id_result:
            st.error("Unable to retrieve user ID. Please ensure the username exists in the database.")
            return

        user_id = user_id_result[0]
    except Exception as e:
        st.error("Failed to retrieve user ID.")
        logging.error(f"Error fetching user ID for {st.session_state['USERNAME']}: {e}")
        return

    # Fetch available data for dropdowns
    try:
        with engine.connect() as conn:
            available_movies_query = text("SELECT title FROM movies")
            available_movies = [row[0] for row in conn.execute(available_movies_query)]

            available_genres = fetch_unique_genres()
            available_directors = fetch_unique_directors()
    except Exception as e:
        st.error("Failed to fetch available options for preferences.")
        logging.error(f"Error fetching preferences data: {e}")
        available_movies, available_genres, available_directors = [], [], []

    # Validate defaults for dropdowns
    valid_favorite_movies = [movie for movie in favorite_movies if movie in available_movies]
    valid_preferred_genres = [genre for genre in preferred_genres if genre in available_genres]
    valid_preferred_directors = [director for director in preferred_directors if director in available_directors]

    # Editable Preferences
    updated_favorite_movies = st.multiselect("Favorite Movies", available_movies, default=valid_favorite_movies)
    updated_preferred_genres = st.multiselect("Preferred Genres", available_genres, default=valid_preferred_genres)
    updated_preferred_styles = st.text_area("Preferred Styles", value=preferred_styles)
    updated_preferred_directors = st.multiselect("Favorite Directors", available_directors, default=valid_preferred_directors)

    if st.button("Save Preferences"):
        try:
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        UPDATE users SET
                            favorite_movies = :favorite_movies,
                            preferred_genres = :preferred_genres,
                            preferred_styles = :preferred_styles,
                            preferred_directors = :preferred_directors
                        WHERE username = :username
                    """),
                    {
                        "favorite_movies": ",".join(updated_favorite_movies),
                        "preferred_genres": ",".join(updated_preferred_genres),
                        "preferred_styles": updated_preferred_styles,
                        "preferred_directors": ",".join(updated_preferred_directors),
                        "username": st.session_state["USERNAME"]
                    }
                )
            st.success("Preferences updated successfully!")
        except Exception as e:
            st.error("Failed to update preferences.")
            logging.error(f"Error updating preferences for {st.session_state['USERNAME']}: {e}")
            
    st.markdown("---")

    # Display Recommendation History
    st.subheader("Your Recommendation History")
    try:
        with engine.connect() as conn:
            rec_history_query = text("""
                SELECT rh.generated_at, m.title, rh.recommendation_type
                FROM recommendation_history rh
                JOIN movies m ON rh.movie_id = m.movie_id
                WHERE rh.user_id = :user_id
                ORDER BY rh.generated_at DESC
            """)
            rec_history = conn.execute(rec_history_query, {"user_id": user_id}).fetchall()
    except Exception as e:
        st.error("Failed to fetch recommendation history.")
        logging.error(f"Error fetching recommendation history for user_id {user_id}: {e}")
        rec_history = []

    if rec_history:
        st.markdown("### Here are the movies we've recommended for you recently:")
        for rec in rec_history:
            st.write(
                f"**{rec[1]}** - Recommended on {rec[0].strftime('%Y-%m-%d %H:%M:%S')} "
                f"- **Type:** {rec[2].capitalize()}"
            )
    else:
        st.warning("No recommendation history found.")

    st.markdown("---")

    # Retrain Interactive Model
    st.subheader("🔄 Retrain Interactive Model")
    if st.button("Retrain Model", key="retrain_model"):
        try:
            with st.spinner("Retraining interactive filtering model..."):
                status = train_interactive_model(engine, retrain=True)
            if status:
                st.success("Interactive model retrained successfully!")
            else:
                st.error("Retraining failed. Check logs for details.")
        except Exception as e:
            st.error("Failed to retrain the interactive model.")
            logging.error(f"Error in retraining interactive model: {e}")
            
    st.markdown("---")   
    
    # Account Management Section
    st.subheader("🔒 Account Management")

    # Styling for buttons
    st.markdown(
        """
        <style>
            .stButton button {
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 5px;
                margin: 0 5px;
            }
            .confirm-button {
                background-color: #d9534f !important;
                color: white !important;
            }
            .cancel-button {
                background-color: #6c757d !important;
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if "action_state" not in st.session_state:
        st.session_state["action_state"] = None  # None, "delete_cache", or "delete_profile"

    # Fetch `user_id` and log it
    try:
        with engine.connect() as conn:
            user_id_query = text("SELECT user_id FROM users WHERE username = :username")
            user_id_result = conn.execute(user_id_query, {"username": st.session_state["USERNAME"]}).fetchone()
        if not user_id_result:
            raise ValueError("User ID not found for the current username.")
        st.session_state["user_id"] = user_id_result[0]
    except Exception as e:
        st.error("Failed to fetch user ID. Ensure your username is valid.")
        logging.error(f"Error fetching user_id: {e}")
        return

    # Function to reset the action state
    def reset_action_state():
        st.session_state["action_state"] = None

    # Function to reset the action state
    def reset_action_state():
        st.session_state["action_state"] = None

    # Render buttons or confirmation dialog
    def render_buttons():
        # Ensure action_state is initialized
        if "action_state" not in st.session_state:
            st.session_state["action_state"] = None

        # Main buttons if no action state is active
        if st.session_state["action_state"] is None:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Delete Profile Cache", key="delete_cache"):
                    st.session_state["action_state"] = "delete_cache"
            with col2:
                if st.button("Delete Profile", key="delete_profile"):
                    st.session_state["action_state"] = "delete_profile"

        if st.session_state["action_state"] == "delete_cache":
            st.warning("Are you sure you want to delete your profile cache? This action cannot be undone.")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Confirm Delete Cache", key="confirm_delete_cache"):
                    # Perform cache deletion logic
                    try:
                        if "user_id" not in st.session_state or not st.session_state["user_id"]:
                            st.error("User ID is missing or invalid in session state.")
                            return

                        user_id = st.session_state["user_id"]

                        # Database operations
                        with engine.begin() as conn:
                            # Delete recommendation history
                            delete_recommendations_result = conn.execute(
                                text("DELETE FROM recommendation_history WHERE user_id = :user_id"),
                                {"user_id": user_id}
                            )
                            st.write(f"Deleted {delete_recommendations_result.rowcount} rows from recommendation_history.")

                            # Nullify specific fields in the users table
                            update_users_result = conn.execute(
                                text("""
                                    UPDATE users
                                    SET preferred_genres = NULL,
                                        preferred_styles = NULL,
                                        favorite_movies = NULL,
                                        preferred_actors = NULL,
                                        preferred_directors = NULL
                                    WHERE user_id = :user_id
                                """),
                                {"user_id": user_id}
                            )
                            st.write(f"Updated {update_users_result.rowcount} rows in users table.")

                        st.success("Your profile cache has been successfully cleared!")
                        st.session_state["action_state"] = None
                        

                    except Exception as e:
                        st.error(f"An unexpected error occurred during deletion: {e}")
            with col2:
                if st.button("Cancel", key="cancel_delete_cache"):
                    reset_action_state()
                    st.experimental_rerun()

        # Delete Profile confirmation
        if st.session_state["action_state"] == "delete_profile":
            st.warning("Are you sure you want to delete your profile? This action will remove all your data permanently.")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Confirm Delete Profile", key="confirm_delete_profile"):
                    try:
                        if "user_id" not in st.session_state or not st.session_state["user_id"]:
                            raise ValueError("User ID is missing or invalid in session state.")

                        st.write(f"Starting profile deletion process for user_id: {st.session_state['user_id']}")

                        # Perform profile deletion
                        with engine.connect() as conn:
                            conn.execute(
                                text("DELETE FROM user_interactions WHERE user_id = :user_id"),
                                {"user_id": st.session_state["user_id"]}
                            )
                            conn.execute(
                                text("DELETE FROM recommendation_history WHERE user_id = :user_id"),
                                {"user_id": st.session_state["user_id"]}
                            )
                            conn.execute(
                                text("DELETE FROM users WHERE user_id = :user_id"),
                                {"user_id": st.session_state["user_id"]}
                            )
                        st.success("Your profile has been successfully deleted!")
                        
                        # Reset action state to return to original button UI
                        st.session_state["action_state"] = None
                        st.experimental_rerun()

                    except Exception as e:
                        st.error(f"Failed to delete profile: {e}")
            with col2:
                if st.button("Cancel", key="cancel_delete_cache"):
                    reset_action_state()
                    st.experimental_rerun()

    # Render buttons
    render_buttons()

# Recommendations Blend
def blend_recommendations(narrative_recommendations, hybrid_recommendations, content_weight, collab_weight, dynamic=False):
    """
    Blend narrative and hybrid recommendations based on specified weights or dynamic adjustment.

    Args:
        narrative_recommendations (list): List of narrative recommendations with 'title' and 'overview'.
        hybrid_recommendations (list): List of hybrid recommendations with 'title', 'overview', and 'score'.
        content_weight (float): Weight assigned to narrative recommendations (manual).
        collab_weight (float): Weight assigned to hybrid recommendations (manual).
        dynamic (bool): If True, dynamically adjust weights based on user feedback.

    Returns:
        list: Blended recommendations sorted by their combined score.
    """
    # Create a dictionary for fast lookup of hybrid recommendation scores
    hybrid_dict = {rec["title"]: rec.get("score", 0) for rec in hybrid_recommendations}

    # Adjust weights dynamically if enabled
    if dynamic:
        # Example of simple dynamic adjustment: Increase weights for narrative if hybrid is underperforming
        feedback_count = len(narrative_recommendations) + len(hybrid_recommendations)
        if feedback_count > 0:
            narrative_feedback_ratio = len(narrative_recommendations) / feedback_count
            hybrid_feedback_ratio = len(hybrid_recommendations) / feedback_count

            content_weight = 0.7 * narrative_feedback_ratio + 0.3
            collab_weight = 0.7 * hybrid_feedback_ratio + 0.3

    # Combine the recommendations
    blended_recommendations = []
    for narrative_rec in narrative_recommendations:
        title = narrative_rec["title"]
        overview = narrative_rec["overview"]

        # Fetch the hybrid score if the title exists in hybrid recommendations
        hybrid_score = hybrid_dict.get(title, 0)

        # Compute the combined score using the weights
        combined_score = (content_weight * 1) + (collab_weight * hybrid_score)

        blended_recommendations.append({
            "title": title,
            "overview": overview,
            "score": combined_score
        })

    # Sort the blended recommendations by their combined score in descending order
    blended_recommendations.sort(key=lambda x: x["score"], reverse=True)

    return blended_recommendations

# Recommendations Tab
def recommendation_tab():
    global_styling()  # Apply global Outfit font styling

    st.markdown(
        """
        <div class="intro-section">
            <h2>🎥 Recommendations</h2>
            <p>Explore movie recommendations tailored to your preferences. Provide feedback to help us refine suggestions!</p>
        </div>
        """, unsafe_allow_html=True
    )

    engine = get_sqlalchemy_engine()

    # Ensure user_id is initialized
    if "user_id" not in st.session_state:
        try:
            with engine.connect() as conn:
                query = text("SELECT user_id FROM users WHERE username = :username")
                result = conn.execute(query, {"username": st.session_state["USERNAME"]}).fetchone()
                if result:
                    st.session_state["user_id"] = result[0]
                else:
                    st.error("User ID not found for the current session. Please log in.")
                    return
        except Exception as e:
            st.error("Failed to fetch user ID. Please contact support.")
            logging.error(f"Error fetching user_id: {e}")
            return

    # Initialize session state
    if "narrative_recommendations" not in st.session_state:
        st.session_state["narrative_recommendations"] = []

    if "hybrid_recommendations" not in st.session_state:
        st.session_state["hybrid_recommendations"] = []

    if "feedback_data" not in st.session_state:
        st.session_state["feedback_data"] = {}

    # Helper Function for Saving User Interactions
    def save_user_interaction(movie_title, interaction_type, rating):
        try:
            with engine.connect() as conn:
                # Fetch movie_id and imdb_id from the database
                movie_query = text("""
                    SELECT movie_id, imdb_id 
                    FROM movies 
                    WHERE title = :title
                """)
                movie_result = conn.execute(movie_query, {"title": movie_title}).fetchone()

                if not movie_result:
                    st.error(f"Movie ID and IMDb ID not found for title: {movie_title}")
                    return

                movie_id, imdb_id = movie_result

                # Insert or update user interaction
                interaction_query = text("""
                    INSERT INTO user_interactions (user_id, movie_id, imdb_id, interaction_type, rating, timestamp)
                    VALUES (:user_id, :movie_id, :imdb_id, :interaction_type, :rating, NOW())
                    ON CONFLICT (user_id, movie_id) DO UPDATE SET
                        interaction_type = EXCLUDED.interaction_type,
                        rating = EXCLUDED.rating,
                        timestamp = EXCLUDED.timestamp
                """)
                conn.execute(
                    interaction_query,
                    {
                        "user_id": st.session_state["user_id"],
                        "movie_id": movie_id,
                        "imdb_id": imdb_id,
                        "interaction_type": interaction_type.lower() if interaction_type != "No Feedback" else None,
                        "rating": rating if interaction_type != "No Feedback" else None,
                    }
                )

            st.success(f"Feedback saved for {movie_title}!")
        except Exception as e:
            st.error(f"Failed to save feedback for {movie_title}.")
            logging.error(f"Error saving feedback for {movie_title}: {e}")

    # Narrative Recommendations
    st.markdown(
        """
        <div class="feature-card">
            <h3>🎭 Narrative Recommendations</h3>
            <p>Describe a scene or theme you like, and we'll recommend movies with similar styles or narratives.</p>
        </div>
        """, unsafe_allow_html=True
    )
    user_scene = st.text_input("Describe a scene or movie theme you like:")
    # Preprocess user input
    if user_scene.strip():
        preprocessed_scene = preprocess_user_input(user_scene)
        logging.info(f"Original Input: {user_scene}")
        logging.info(f"Preprocessed Input: {preprocessed_scene}")
    else:
        preprocessed_scene = ""
        
    # Hybrid Recommender Toggle
    hybrid_recommender = st.checkbox("Enable Hybrid Recommender", key="hybrid_recommender")

    # Generate Narrative Recommendations
    if not hybrid_recommender and st.button("Get Narrative Recommendations", key="narrative_recommendations_btn"):
        if not user_scene.strip():
            st.warning("Please describe a scene or theme before generating recommendations.")
        else:
            try:
                with st.spinner("Generating narrative recommendations..."):
                    with engine.connect() as connection:
                        query = text("""
                            SELECT imdb_id, genre
                            FROM movies
                        """)
                        scene_data = pd.read_sql_query(query, connection)

                    if scene_data.empty:
                        st.error("Failed to fetch scene data for evaluation.")
                        logging.error("Scene data fetched is empty.")
                        return
                    raw_recommendations = generate_narrative_recommendations(engine, user_scene)
                    seen_titles = set()

                    # Process recommendations
                    recommendations = []
                    for rec in raw_recommendations:
                        if rec["title"] not in seen_titles:
                            seen_titles.add(rec["title"])
                            recommendations.append(rec)

                            # Fetch movie_id for the recommendation
                            movie_id = get_movie_id_by_title(rec["title"])
                            if movie_id:
                                # Log the recommendation to the database
                                log_recommendation(
                                    user_id=st.session_state["user_id"],
                                    movie_id=movie_id,
                                    recommendation_type="narrative"
                                )
                            else:
                                logging.warning(f"Movie ID not found for title: {rec['title']}")

                            # Initialize feedback data
                            st.session_state["feedback_data"].setdefault(
                                rec["title"], {"interaction_type": None, "rating": None}
                            )

                    # Save recommendations to session state
                    st.session_state["narrative_recommendations"] = recommendations

                st.success("Recommendations generated successfully!")
                # Do not display recommendations here

                # Metadata for evaluation
                movie_metadata = {row["imdb_id"]: row["genre"] for _, row in scene_data.iterrows()}

                # Perform evaluations
                similarity_metrics = similarity_analysis(recommendations)
                # metadata_metrics = metadata_evaluation(user_scene, recommendations, movie_metadata)
                
                # Display Metrics
                st.markdown("### Evaluation Metrics")
                st.write(f"**Recall:** {similarity_metrics['recall']:.2f}")
                st.write(f"**F1-Score:** {similarity_metrics['f1_score']:.2f}")
                st.write(f"**Mean Similarity:** {(similarity_metrics['mean_similarity']):.2f}")
                st.write(f"**Precision Above Threshold:** {similarity_metrics['precision_above_threshold']:.2f}")
                st.write(f"**Above Threshold Count:** {similarity_metrics['above_threshold_count']}")
                # st.write(f"**Metadata Metrics:** {metadata_metrics}")

            except Exception as e:
                st.error("Failed to generate narrative recommendations.")
                logging.error(f"Error in narrative recommendations: {e}")

    # Display Narrative Recommendations
    if not hybrid_recommender and st.session_state["narrative_recommendations"]:
        st.markdown(
            """
            <div class="quiz-section">
                <h4 style="color: #FFAA00;">Recommended Movies:</h4>
            </div>
            """, unsafe_allow_html=True
        )
        for idx, rec in enumerate(st.session_state["narrative_recommendations"]):
            with st.container():
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <h4>🎬 {rec["title"]}</h4>
                        <p>{rec['overview']} (Score: {(rec.get('similarity', 0)):.2f})</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                interaction_type = st.radio(
                    f"Did you like {rec['title']}?",
                    options=["Like", "Dislike", "No Feedback"],
                    index=2 if st.session_state["feedback_data"][rec["title"]]["interaction_type"] is None else
                    ["Like", "Dislike", "No Feedback"].index(st.session_state["feedback_data"][rec["title"]]["interaction_type"]),
                    key=f"interaction_{rec['title']}"
                )
                rating = st.slider(
                    f"Rate {rec['title']}:",
                    min_value=0,
                    max_value=10,
                    step=1,
                    value=st.session_state["feedback_data"][rec["title"]]["rating"] or 0,
                    key=f"rating_{rec['title']}"
                )
                st.session_state["feedback_data"][rec["title"]].update({"interaction_type": interaction_type, "rating": rating})
                if st.button(f"Save Feedback for {rec['title']}", key=f"save_feedback_{rec['title']}"):
                    save_user_interaction(rec["title"], interaction_type, rating)

    # Hybrid Recommendations Section
    if hybrid_recommender:
        st.markdown(
            """
            <div class="feature-card">
                <h3>⚙️ Hybrid Recommendations</h3>
                <p>Fine-tune the recommendation engine by adjusting the balance between content-based and collaborative filtering.</p>
            </div>
            """, unsafe_allow_html=True
        )

        # Sliders for weight adjustments
        st.markdown(
            """
            <div class="quiz-section">
                <h4 style="color: #FFAA00;">Adjust Weight Parameters:</h4>
            </div>
            """, unsafe_allow_html=True
        )
        content_weight = st.slider("Content Weight", 0.0, 1.0, 0.4, key="content_weight")
        collab_weight = st.slider("Collaborative Weight", 0.0, 1.0, 0.6, key="collab_weight")

        # Generate and Display Hybrid Recommendations with Pagination
        if st.button("Get Hybrid Recommendations", key="hybrid_recommendations_btn"):
            try:
                with st.spinner("Generating hybrid recommendations..."):
                    # Clear narrative recommendations for hybrid mode
                    st.session_state["narrative_recommendations"] = []
                    raw_hybrid_recommendations = generate_hybrid_recommendations(
                        engine, user_scene, st.session_state["user_id"], content_weight, collab_weight
                    )
                    seen_titles = set()
                    st.session_state["hybrid_recommendations"] = []

                    for rec in raw_hybrid_recommendations:
                        if rec["title"] not in seen_titles:
                            seen_titles.add(rec["title"])
                            st.session_state["hybrid_recommendations"].append(rec)
                            
                            # Fetch movie_id and log the recommendation
                            movie_id = get_movie_id_by_title(rec["title"])
                            if movie_id:
                                log_recommendation(
                                    user_id=st.session_state["user_id"],
                                    movie_id=movie_id,
                                    recommendation_type="hybrid"
                                )
                            else:
                                logging.warning(f"Movie ID not found for title: {rec['title']}")

                            # Initialize feedback data
                            st.session_state["feedback_data"].setdefault(
                                rec["title"], {"interaction_type": None, "rating": None}
                            )
                    
                # Metadata for evaluation
                with engine.connect() as connection:
                    query = text("""
                        SELECT imdb_id, genre
                        FROM movies
                    """)
                    scene_data = pd.read_sql_query(query, connection)
                movie_metadata = {row["imdb_id"]: row["genre"] for _, row in scene_data.iterrows()}

                # Perform evaluations
                similarity_metrics = similarity_analysis(st.session_state["hybrid_recommendations"])
                # metadata_metrics = metadata_evaluation(user_scene, st.session_state["hybrid_recommendations"], movie_metadata)
                
                # Display Metrics
                st.markdown("### Evaluation Metrics")
                st.write(f"**Recall:** {similarity_metrics['recall']:.2f}")
                st.write(f"**F1-Score:** {similarity_metrics['f1_score']:.2f}")
                st.write(f"**Mean Similarity:** {similarity_metrics['mean_similarity']:.2f}")
                st.write(f"**Precision Above Threshold:** {similarity_metrics['precision_above_threshold']:.2f}")
                st.write(f"**Above Threshold Count:** {similarity_metrics['above_threshold_count']}")
                # st.write(f"**Metadata Metrics:** {metadata_metrics}")
                                   
            except Exception as e:
                st.error("Failed to generate hybrid recommendations.")
                logging.error(f"Error in hybrid recommendations: {e}")

        # Display Recommendations with Pagination
        if st.session_state["hybrid_recommendations"]:
            items_per_page = 5
            total_items = len(st.session_state["hybrid_recommendations"])
            total_pages = (total_items + items_per_page - 1) // items_per_page  # Ceiling division
            current_page = st.session_state.get("current_page", 1)

            # Ensure current_page is within bounds
            current_page = max(1, min(current_page, total_pages))
            st.session_state["current_page"] = current_page

            start_index = (current_page - 1) * items_per_page
            end_index = start_index + items_per_page

            # Display Recommendations for Current Page
            for rec in st.session_state["hybrid_recommendations"][start_index:end_index]:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <h4>🎬 {rec['title']}</h4>
                        <p>{rec['overview']} (Score: {rec.get('similarity', 0):.2f})</p>
                    </div>
                    """, unsafe_allow_html=True
                )

                # User Feedback Section
                interaction_type = st.radio(
                    f"Did you like {rec['title']}?",
                    options=["Like", "Dislike", "No Feedback"],
                    index=2 if st.session_state["feedback_data"][rec["title"]]["interaction_type"] is None else
                    ["Like", "Dislike", "No Feedback"].index(st.session_state["feedback_data"][rec["title"]]["interaction_type"]),
                    key=f"interaction_hybrid_{rec['title']}"
                )
                rating = st.slider(
                    f"Rate {rec['title']}:",
                    min_value=0,
                    max_value=10,
                    step=1,
                    value=st.session_state["feedback_data"][rec["title"]]["rating"] or 0,
                    key=f"rating_hybrid_{rec['title']}"
                )
                st.session_state["feedback_data"][rec["title"]].update({"interaction_type": interaction_type, "rating": rating})
                if st.button(f"Save Feedback for {rec['title']}", key=f"save_feedback_hybrid_{rec['title']}"):
                    save_user_interaction(rec["title"], interaction_type, rating)

            # Pagination Controls
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if current_page > 1:
                    if st.button("Previous", key="prev_page"):
                        st.session_state["current_page"] = current_page - 1
            with col2:
                st.markdown(f"<p style='text-align: center;'>Page {current_page} of {total_pages}</p>", unsafe_allow_html=True)
            with col3:
                if current_page < total_pages:
                    if st.button("Next", key="next_page"):
                        st.session_state["current_page"] = current_page + 1

    st.markdown("---")

    # Recommendation Quiz Section
    st.markdown(
        """
        <div class="feature-card">
            <h3>📝 Recommendation Quiz</h3>
            <p style="font-size: 16px; line-height: 1.6;">
                Answer a couple of quick questions, and we'll recommend movies tailored to your mood, preferences, and favorite genre.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    try:
        genres = fetch_unique_genres()
        if not genres:
            st.error("No genres available. Please check the database connection.")
        else:
            # Quiz Section: Mood
            st.markdown(
                """
                <div class="quiz-card">
                    <h4 style="color: #FFAA00; margin-bottom: 10px;">1. What's your current mood?</h4>
                    <p style="color: #BBBBBB; font-size: 14px; margin-bottom: 15px;">
                        Select a mood that matches your current state of mind.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            mood = st.selectbox(
                "Select your current mood:",
                [
                    "Adventurous", "Romantic", "Thought-Provoking", "Relaxing",
                    "Suspenseful", "Action-Packed", "Fantasy", "Horror",
                    "Epic", "Sci-Fi", "Whimsical", "Dramatic", "Inspirational"
                ],
                key="mood_selectbox"
            )

            # Quiz Section: Genre
            st.markdown(
                """
                <div class="quiz-card">
                    <h4 style="color: #FFAA00; margin-bottom: 10px;">2. Choose your favorite genre:</h4>
                    <p style="color: #BBBBBB; font-size: 14px; margin-bottom: 15px;">
                        Select a genre that you enjoy the most.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            favorite_genre = st.selectbox(
                "Select your favorite genre:",
                genres,
                key="genre_selectbox"
            )

            # Quiz Section: Preferred Runtime
            st.markdown(
                """
                <div class="quiz-card">
                    <h4 style="color: #FFAA00; margin-bottom: 10px;">3. How long do you prefer your movies to be?</h4>
                    <p style="color: #BBBBBB; font-size: 14px; margin-bottom: 15px;">
                        Select your preferred runtime range.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            runtime = st.selectbox(
                "Select your preferred runtime:",
                ["< 90 minutes", "90-120 minutes", "120-150 minutes", "> 150 minutes"],
                key="runtime_selectbox"
            )

            # Quiz Section: Release Year Range
            st.markdown(
                """
                <div class="quiz-card">
                    <h4 style="color: #FFAA00; margin-bottom: 10px;">4. What is your preferred release year range?</h4>
                    <p style="color: #BBBBBB; font-size: 14px; margin-bottom: 15px;">
                        Select a release year range that suits your preference.
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            year = st.slider(
                "Select your preferred release year range:",
                min_value=1950,
                max_value=2025,
                value=(2000, 2015),
                key="year_slider"
            )

            # Submit Button
            if st.button("Submit Quiz", key="quiz_recommendations"):
                try:
                    with st.spinner("Fetching recommendations based on your quiz..."):
                        recommendations = quiz_based_recommendations(
                            mood, favorite_genre, runtime, year
                        )

                    if recommendations:
                        st.markdown("<h4 style='margin-top: 20px;'>Your Movie Recommendations:</h4>", unsafe_allow_html=True)
                        for rec in recommendations:
                            st.markdown(
                                f"""
                                <div class="feature-card" style="margin-bottom: 20px;">
                                    <h4>🎬 {rec['title']}</h4>
                                    <p>{rec['overview']} (Genre: {rec['genre']})</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # Log the recommendation to the database
                            movie_id = get_movie_id_by_title(rec["title"])
                            if movie_id:
                                log_recommendation(
                                    user_id=st.session_state["user_id"],
                                    movie_id=movie_id,
                                    recommendation_type="quiz"
                                )
                            else:
                                logging.warning(f"Movie ID not found for title: {rec['title']}")

                            # Initialize feedback data
                            st.session_state["feedback_data"].setdefault(
                                rec["title"], {"interaction_type": None, "rating": None}
                            )
                    else:
                        st.warning("No quiz-based recommendations available for the selected preferences.")
                except Exception as e:
                    st.error("Failed to generate quiz-based recommendations.")
                    logging.error(f"Error in quiz-based recommendations: {e}")

    except Exception as e:
        st.error("Failed to fetch genres for the quiz.")
        logging.error(f"Error fetching genres: {e}")

# Performance and Analytics Tab
def performance_analytics_tab():
    global_styling()

    # Intro Section
    st.markdown(
        """
        <div class="intro-section">
            <h2>🔢 Performance and Analytics</h2>
            <p style="font-size: 16px; line-height: 1.6;">
                Discover how the system performs, explore user interactions, and track feedback trends—all presented in clear, engaging visuals to help you make informed decisions effortlessly.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    engine = get_sqlalchemy_engine()
    if not engine:
        st.error("Database connection failed. Please check the connection.")
        return

    # Section 1: Model Performance Metrics
    st.markdown(
        """
        <div class="feature-card">
            <h3 style="margin: 0; font-size: 20px;">📈 Model Performance Metrics</h3>
            <p style="font-size: 14px; color: #B0B0B0;">Track the performance of your recommendation model over time, including precision, recall, F1-score, accuracy, RMSE, MAE, coverage, engagement rate, and number of movies trained on.</p>
        </div>
        """, unsafe_allow_html=True
    )
    try:
        user_id = st.session_state.get("user_id")
        if not user_id:
            st.warning("User ID is missing from the session state.")
            return

        metrics = fetch_model_metrics(engine, user_id=user_id)

        if metrics.empty:
            st.warning("No model performance data available.")
        else:
            # Plot Key Metrics Over Time
            st.markdown("<h4>Performance Metrics Over Time:</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot multiple metrics
            ax.plot(metrics["timestamp"], metrics["precision"], label="Precision", marker="o")
            ax.plot(metrics["timestamp"], metrics["recall"], label="Recall", marker="x")
            ax.plot(metrics["timestamp"], metrics["f1_score"], label="F1 Score", marker="s")
            if "accuracy" in metrics:
                ax.plot(metrics["timestamp"], metrics["accuracy"], label="Accuracy", marker="^")
            ax.plot(metrics["timestamp"], metrics["rmse"], label="RMSE", marker="v")
            ax.plot(metrics["timestamp"], metrics["mae"], label="MAE", marker="d")
            ax.plot(metrics["timestamp"], metrics["coverage"], label="Coverage", marker="p")
            if "engagement_rate" in metrics:
                ax.plot(metrics["timestamp"], metrics["engagement_rate"], label="Engagement Rate", marker="h")

            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Metric Value")
            ax.set_title("Model Performance Metrics Over Time")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Display Raw Metrics Table
            st.markdown("<h4>Raw Metrics Data:</h4>", unsafe_allow_html=True)
            st.dataframe(metrics)

            # Show Summary of Metrics
            st.markdown("<h4>Summary of Latest Metrics:</h4>", unsafe_allow_html=True)

            # Ensure safe handling of metrics
            latest_metrics = metrics.iloc[-1]

            def safe_format(value, fmt="{:.2f}"):
                """Safely format numeric values, handling None and NaN."""
                try:
                    if pd.isna(value):
                        return "N/A"
                    return fmt.format(value)
                except (ValueError, TypeError):
                    return "N/A"

            st.markdown(f"""
                - **Timestamp:** {latest_metrics['timestamp']}
                - **Precision:** {safe_format(latest_metrics['precision'])}
                - **Recall:** {safe_format(latest_metrics['recall'])}
                - **F1 Score:** {safe_format(latest_metrics['f1_score'])}
                - **Accuracy:** {safe_format(latest_metrics.get('accuracy', None))}
                - **RMSE:** {safe_format(latest_metrics['rmse'])}
                - **MAE:** {safe_format(latest_metrics['mae'])}
                - **Coverage:** {safe_format(latest_metrics['coverage'])}
                - **Engagement Rate:** {safe_format(latest_metrics.get('engagement_rate', None))}
                - **Number of Movies Trained:** {latest_metrics['num_movies'] if 'num_movies' in latest_metrics else 'N/A'}
                - **Notes:** {latest_metrics['notes'] if 'notes' in latest_metrics else 'N/A'}
            """)
    except Exception as e:
        st.error("Failed to fetch or display model performance metrics.")
        logging.error(f"Error in performance_analytics_tab - performance section: {e}")

    st.markdown("---")

    # Section 2: User Interaction Insights
    st.markdown(
        """
        <div class="feature-card">
            <h3 style="margin: 0; font-size: 20px;">👤 User Interaction Insights</h3>
            <p style="font-size: 14px; color: #B0B0B0;">Explore your interaction trends and see how your preferences align with genres and activity patterns. Gain insights into what makes your experience unique!</p>
        </div>
        """, unsafe_allow_html=True
    )
    try:
        with engine.connect() as conn:
            user_id_query = text("SELECT user_id FROM users WHERE username = :username")
            user_id_result = conn.execute(user_id_query, {"username": st.session_state["USERNAME"]}).fetchone()

        if not user_id_result:
            st.error("Unable to retrieve user ID. Please ensure the username exists in the database.")
            return

        user_id = user_id_result[0]

        insights = fetch_user_insights(user_id)

        if insights:
            from collections import Counter
            genre_counter = Counter()
            for insight in insights:
                genres = insight["genre"].split(", ")
                genre_counter.update(genres)

            # Genre Insights Table
            genre_data = pd.DataFrame.from_dict(genre_counter, orient="index", columns=["Count"])
            genre_data = genre_data.reset_index().rename(columns={"index": "Genre"})
            genre_data = genre_data.sort_values(by="Count", ascending=False)

            # Summary
            st.markdown("<h4>Summary:</h4>", unsafe_allow_html=True)
            total_interactions = genre_data["Count"].sum()
            top_genre = genre_data.iloc[0]["Genre"]
            st.markdown(f"**Total Interactions**: {total_interactions}")
            st.markdown(f"**Top Genre**: {top_genre} ({genre_data.iloc[0]['Count']} interactions)")

            # Genre Distribution Chart
            st.markdown("<h4>Genre Distribution:</h4>", unsafe_allow_html=True)
            st.bar_chart(genre_data.set_index("Genre"))

            # Detailed Genre Insights
            st.markdown("<h4>Detailed Genre Insights:</h4>", unsafe_allow_html=True)
            for idx, row in genre_data.iterrows():
                st.write(f"**Genre**: {row['Genre']}, **Count**: {row['Count']}")

        else:
            st.warning("No interaction data available.")

    except Exception as e:
        st.error("An error occurred while fetching user interaction insights.")
        logging.error(f"Error in performance_analytics_tab - analytics section: {e}")

    st.markdown("---")

    # Section 3: Interactive Feedback Overview
    st.markdown(
        """
        <div class="feature-card">
            <h3 style="margin: 0; font-size: 20px;">💬 Interactive Feedback Overview</h3>
            <p style="font-size: 14px; color: #B0B0B0;">View your feedback trends and dive into your likes, dislikes, and ratings to see how they shape personalized recommendations just for you!</p>
        </div>
        """, unsafe_allow_html=True
    )
    try:
        feedback_query = text("""
            SELECT m.title, ui.interaction_type AS interact, ui.rating, ui.timestamp
            FROM user_interactions ui
            JOIN movies m ON ui.movie_id = m.movie_id
            WHERE ui.user_id = :user_id
            ORDER BY ui.timestamp DESC
            LIMIT 20
        """)
        with engine.connect() as conn:
            feedback_data = pd.read_sql_query(feedback_query, conn, params={"user_id": user_id})

        if not feedback_data.empty:
            # Feedback Summary
            st.markdown("<h4>Summary:</h4>", unsafe_allow_html=True)
            total_feedback = len(feedback_data)
            likes_count = len(feedback_data[feedback_data["interact"] == "like"])
            dislikes_count = len(feedback_data[feedback_data["interact"] == "dislike"])
            average_rating = feedback_data["rating"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Feedback", total_feedback)
            col2.metric("Likes", likes_count, delta=f"{likes_count - dislikes_count}" if not pd.isna(likes_count) and not pd.isna(dislikes_count) else "N/A")
            col3.metric("Average Rating", f"{average_rating:.2f}" if not pd.isna(average_rating) else "N/A")

            # Feedback Table
            st.markdown("<h4>Recent Feedback:</h4>", unsafe_allow_html=True)
            feedback_data["rating"] = feedback_data["rating"].fillna(0)  # Replace NaN ratings with 0 for formatting
            st.dataframe(feedback_data.style.format({"rating": "{:.1f}"}))

        else:
            st.warning("No feedback data available.")

    except Exception as e:
        st.error("Failed to fetch user feedback data.")
        logging.error(f"Error in performance_analytics_tab - feedback section: {e}")

# Recommendation Quiz
def quiz_recommendation():
    """
    Provide a quiz-based recommendation system based on the user's mood and genre preference.
    """
    st.title("Recommendation Quiz")

    # Fetch genres dynamically
    try:
        genres = fetch_unique_genres()
        if not genres:
            st.error("No genres available. Please check the database connection.")
            return
    except Exception as e:
        st.error("Failed to load genres for the quiz.")
        logging.error(f"Error in quiz_recommendation - genres: {e}")
        return

    # Comprehensive mood list with descriptions
    moods = {
        "Adventurous": "Thrilling, exciting, and dangerous experiences.",
        "Romantic": "Love, passion, and relationships.",
        "Thought-Provoking": "Philosophical, inspiring, and mysterious themes.",
        "Relaxing": "Comedy, light-hearted, and family-friendly.",
        "Suspenseful": "Mystery, thriller, and crime stories.",
        "Action-Packed": "War, chase, and fight scenes.",
        "Fantasy": "Magical, mythical, and otherworldly adventures.",
        "Dramatic": "Emotional, intense, and heartfelt narratives.",
        "Dark": "Moody, psychological, and gothic.",
        "Epic": "Grand, large-scale, and historical stories.",
        "Lighthearted": "Funny, wholesome, and feel-good themes.",
        "Inspirational": "Heroic, uplifting, and motivational stories.",
        "Scary": "Horror, paranormal, and spine-chilling tales.",
        "Sci-Fi": "Futuristic, technological, and space adventures.",
        "Whimsical": "Imaginative, playful, and childlike themes."
    }

    # Display the mood options with descriptions
    selected_mood = st.radio(
        "How are you feeling right now? Choose the best match:",
        options=list(moods.keys()),
        format_func=lambda mood: f"{mood} ({moods[mood]})",  # Show description alongside the option
        index=0
    )

    # Allow users to enter a custom mood (optional)
    custom_mood = st.text_input("Or describe your own mood here (optional):")

    # Combine predefined mood with the custom mood if provided
    final_mood = f"{selected_mood} - {custom_mood}" if custom_mood.strip() else selected_mood

    # Fetch genre selection
    favorite_genre = st.selectbox("Choose your favorite genre", genres)

    # Fetch quiz-based recommendations
    if st.button("Get Quiz-Based Recommendations"):
        try:
            engine = get_sqlalchemy_engine()
            with st.spinner("Fetching recommendations based on your quiz inputs..."):
                recommendations = quiz_based_recommendations(final_mood, favorite_genre)
            if recommendations:
                for rec in recommendations:
                    st.write(f"**{rec['title']}** - {rec['overview']} (Genre: {rec['genre']})")
            else:
                st.warning("No recommendations available for the selected mood and genre.")
        except Exception as e:
            st.error("Failed to generate quiz-based recommendations.")
            logging.error(f"Error in quiz_recommendation: {e}")