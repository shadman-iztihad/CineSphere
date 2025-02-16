import os
import sys
import pickle
import logging
import pandas as pd
from math import sqrt
import streamlit as st
from sqlalchemy.sql import text
from surprise import Dataset, Reader, SVDpp
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add relevant folders to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
components_path = os.path.join(project_root, "components")
recommendation_path = os.path.join(project_root, "recommendation")

sys.path.insert(0, components_path)
sys.path.insert(0, recommendation_path)

from components.utils import get_sqlalchemy_engine, log_model_metrics

# Configure logging
logging.basicConfig(
    filename="interaction_handler.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

model_path = os.path.abspath("interactive_model.pkl")
metadata_path = os.path.abspath("interactive_model_metadata.pkl")

logging.debug(f"Model path: {model_path}")
logging.debug(f"Metadata path: {metadata_path}")

# Fetch Users
def fetch_users(engine):
    """
    Fetch user IDs and usernames from the database.

    Args:
        engine: SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: DataFrame containing user_id and username.
    """
    try:
        query = text("SELECT user_id, username FROM users")
        with engine.connect() as connection:
            result = connection.execute(query)
            users = pd.DataFrame(result.fetchall(), columns=result.keys())
        logging.info(f"Fetched {len(users)} users from the database.")
        return users
    except Exception as e:
        logging.error(f"Error fetching users: {e}")
        return pd.DataFrame()

# Fetch User Profile
def fetch_user_profile(username):
    """
    Fetch the name and email of a user by their username.

    Args:
        username (str): The username of the user.

    Returns:
        dict: A dictionary containing 'name' and 'email' if the user exists, otherwise an empty dictionary.
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logging.error("Failed to establish database connection.")
        return {}

    try:
        query = text("SELECT name, email FROM users WHERE username = :username")
        with engine.connect() as connection:
            result = connection.execute(query, {"username": username}).fetchone()
        return {"name": result["name"], "email": result["email"]} if result else {}
    except Exception as e:
        logging.error(f"Error fetching user profile for username '{username}': {e}")
        return {}

# Fetch Movies
def fetch_movies(engine):
    """
    Fetch all movies and their metadata from the database.

    Args:
        engine: SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: DataFrame containing movie_id, imdb_id, title, genre, overview, and director.
    """
    try:
        query = text("""
            SELECT movie_id, imdb_id, title, genre, overview, director
            FROM movies
        """)
        with engine.connect() as connection:
            result = connection.execute(query)
            movies = pd.DataFrame(result.fetchall(), columns=result.keys())
        logging.info(f"Fetched {len(movies)} movies from the database.")
        return movies
    except Exception as e:
        logging.error(f"Error fetching movies: {e}")
        return pd.DataFrame()

# Fetch User Insights
def fetch_user_insights(user_id):
    """
    Fetch user interaction insights based on user_id.

    Args:
        user_id (int): User ID.

    Returns:
        list of dict: Insights grouped by genre with their interaction counts.
    """
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT genre, COUNT(*) AS count
                FROM user_interactions
                JOIN movies ON user_interactions.movie_id = movies.movie_id
                WHERE user_interactions.user_id = :user_id
                GROUP BY genre
                ORDER BY count DESC
            """)
            results = conn.execute(query, {"user_id": user_id}).fetchall()

        return [{"genre": row[0], "count": row[1]} for row in results]
    except Exception as e:
        logging.error(f"Error fetching user insights for user_id {user_id}: {e}")
        return []

# Log Interactions
def log_interactions(engine, interactions_data):
    try:
        with engine.begin() as connection:
            for user_id, interactions in interactions_data.items():
                for interaction in interactions:
                    movie_id = interaction["movie_id"]
                    interaction_type = interaction["interaction_type"]
                    rating = interaction.get("rating", None)

                    # Check for existing interaction
                    check_query = text("""
                        SELECT COUNT(*)
                        FROM user_interactions
                        WHERE user_id = :user_id AND movie_id = :movie_id
                    """)
                    existing_count = connection.execute(check_query, {"user_id": user_id, "movie_id": movie_id}).scalar()

                    # Insert interaction only if it doesn't already exist
                    if existing_count == 0:
                        insert_query = text("""
                            INSERT INTO user_interactions (user_id, movie_id, interaction_type, rating)
                            VALUES (:user_id, :movie_id, :interaction_type, :rating)
                            ON CONFLICT (user_id, movie_id)
                            DO UPDATE SET interaction_type = :interaction_type, rating = :rating
                        """)
                        connection.execute(insert_query, {
                            "user_id": user_id,
                            "movie_id": movie_id,
                            "interaction_type": interaction_type,
                            "rating": rating
                        })
            logging.info("Logged interactions for users.")
    except Exception as e:
        logging.error(f"Error logging interactions: {e}")

# Log Recommendations
def log_recommendation(user_id, movie_id, recommendation_type):
    """
    Log a recommendation into the recommendation_history table.

    Args:
        user_id (int): The ID of the user.
        movie_id (int): The ID of the recommended movie.
        recommendation_type (str): The type of recommendation ('quiz', 'hybrid', etc.).
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logging.error("Failed to establish database connection.")
        return

    try:
        query = text("""
            INSERT INTO recommendation_history (user_id, movie_id, recommendation_type)
            VALUES (:user_id, :movie_id, :recommendation_type)
            ON CONFLICT (user_id, movie_id, recommendation_type) DO NOTHING
        """)
        with engine.connect() as conn:
            conn.execute(query, {
                "user_id": user_id,
                "movie_id": movie_id,
                "recommendation_type": recommendation_type
            })
        logging.info(f"Logged recommendation for user_id={user_id}, movie_id={movie_id}, type={recommendation_type}")
    except Exception as e:
        logging.error(f"Failed to log recommendation: {e}")

# Function to fetch quiz-based recommendations
def quiz_based_recommendations(mood, genre, runtime=None, year=None):
    """
    Generate quiz-based recommendations using mood, genre, runtime, and release year.

    Args:
        mood (str): User's mood.
        genre (str): User's favorite genre.
        runtime (str, optional): Preferred runtime range (e.g., "< 90 minutes").
        year (tuple, optional): Preferred release year range (start, end).

    Returns:
        list: List of recommended movies with metadata.
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logging.error("Failed to establish database connection.")
        return []

    if not mood or not genre:
        logging.warning("Invalid mood or genre provided for quiz-based recommendations.")
        return []

    # Expanded mood keywords
    mood_keywords = {
        "Adventurous": [
            "adventure", "thrill", "danger", "explore", "journey", "expedition", 
            "quest", "trek", "voyage", "wanderlust", "wild", "exploration"
        ],
        "Romantic": [
            "love", "romance", "relationship", "passion", "intimacy", "wedding", 
            "proposal", "heartwarming", "affection", "courtship", "charming", "soulmates"
        ],
        "Thought-Provoking": [
            "philosophy", "inspiring", "mystery", "psychological", "existential", 
            "intellectual", "thoughtful", "reflective", "contemplative", "moral dilemma", 
            "mind-bending", "provocative", "philosophical"
        ],
        "Relaxing": [
            "comedy", "family", "light-hearted", "funny", "wholesome", "uplifting", 
            "feel-good", "humorous", "chill", "laid-back", "carefree", "soothing", 
            "heartwarming"
        ],
        "Suspenseful": [
            "mystery", "thriller", "crime", "detective", "intrigue", "tense", "cliffhanger", 
            "whodunit", "paranoia", "conspiracy", "espionage", "undercover", "chilling"
        ],
        "Action-Packed": [
            "action", "war", "battle", "chase", "fight", "explosive", "intense", 
            "combat", "heroic", "showdown", "military", "adrenaline", "epic fights", 
            "stunts", "brawls"
        ],
        "Fantasy": [
            "magic", "supernatural", "myth", "fairy tale", "wizards", "dragons", 
            "sorcery", "mythical", "enchanted", "otherworldly", "legendary", 
            "otherworlds", "imaginative", "fantastical", "adventure in a magical realm"
        ],
        "Horror": [
            "scary", "paranormal", "haunted", "supernatural", "fear", "ghosts", 
            "creepy", "spooky", "eerie", "terror", "monsters", "psychological horror", 
            "jump scares", "haunting"
        ],
        "Epic": [
            "grand", "historical", "scale", "legend", "mythology", "dynasty", 
            "conquer", "ruler", "empires", "adventure saga", "heroic tale", 
            "victory", "historical events"
        ],
        "Sci-Fi": [
            "space", "technology", "future", "robots", "aliens", "artificial intelligence", 
            "time travel", "dystopian", "futuristic", "galaxy", "cyberpunk", 
            "outer space", "utopian", "interstellar"
        ],
        "Whimsical": [
            "imaginative", "playful", "childlike", "quirky", "fun", "creative", 
            "fantastical", "dreamlike", "light-hearted fantasy", "cartoonish", 
            "whimsy", "magical", "colorful", "charming"
        ],
        "Dramatic": [
            "intense", "emotional", "heartfelt", "tragic", "conflict", "family struggles", 
            "tearjerker", "melodrama", "powerful", "gripping", "character-driven", 
            "narrative depth", "moving"
        ],
        "Inspirational": [
            "heroic", "uplifting", "motivational", "triumph", "resilience", 
            "overcoming odds", "inspiring", "hope", "achievements", "redemption", 
            "transformational", "dreams", "perseverance"
        ]
    }

    try:
        # Retrieve keywords for the selected mood
        keywords = mood_keywords.get(mood, [])
        if not keywords:
            logging.warning(f"No keywords found for mood: {mood}")
            return []

        # Generate dynamic SQL conditions for keywords
        conditions = " OR ".join(
            [f"(keywords ILIKE :kw_{i} OR overview ILIKE :kw_{i})" for i in range(len(keywords))]
        )

        # Runtime filter
        runtime_condition = ""
        if runtime:
            if runtime == "< 90 minutes":
                runtime_condition = "AND runtime < 90"
            elif runtime == "90-120 minutes":
                runtime_condition = "AND runtime BETWEEN 90 AND 120"
            elif runtime == "120-150 minutes":
                runtime_condition = "AND runtime BETWEEN 120 AND 150"
            elif runtime == "> 150 minutes":
                runtime_condition = "AND runtime > 150"

        # Release year filter
        year_condition = ""
        if year:
            year_condition = f"AND year BETWEEN :start_year AND :end_year"

        query = text(f"""
            SELECT title, genre, overview, keywords, runtime, year
            FROM movies
            WHERE ({conditions}) AND genre ILIKE :genre
            {runtime_condition} {year_condition}
            ORDER BY RANDOM()
            LIMIT 5
        """)

        # Prepare query parameters
        params = {f"kw_{i}": f"%{word}%" for i, word in enumerate(keywords)}
        params["genre"] = f"%{genre}%"
        if year:
            params["start_year"], params["end_year"] = year

        # Execute query
        with engine.connect() as conn:
            results = conn.execute(query, params).fetchall()

        # Format results
        return [{"title": row["title"], "genre": row["genre"], "overview": row["overview"]} for row in results]
    except Exception as e:
        logging.error(f"Error fetching quiz-based recommendations: {e}")
        return []

# # Function to fetch interaction data
# def fetch_interaction_data(engine):
#     """
#     Fetch interaction data from the database for interactive filtering.

#     Args:
#         engine: SQLAlchemy engine for database connection.

#     Returns:
#         pd.DataFrame: A DataFrame containing user_id, imdb_id, and rating.
#     """
#     try:
#         query = text("""
#             SELECT user_id, imdb_id, COALESCE(rating, 0) AS rating
#             FROM user_interactions
#             WHERE rating IS NOT NULL
#         """)
#         with engine.connect() as connection:
#             result = connection.execute(query)
#             interactions = pd.DataFrame(result.fetchall(), columns=result.keys())
#         logging.info(f"Fetched {len(interactions)} interactions from the database.")
#         return interactions
#     except Exception as e:
#         logging.error(f"Error fetching interaction data: {e}")
#         return pd.DataFrame()

def evaluate_model(predictions, true_ratings, interaction_types):
    """
    Evaluate the model's performance using interaction types for relevance.

    Args:
        predictions (list): Predicted ratings.
        true_ratings (list): Actual ratings.
        interaction_types (list): List of interaction types ("like" or "dislike").

    Returns:
        dict: Dictionary of calculated metrics.
    """
    # Convert interaction types to relevance labels
    relevant_indices = [i for i, interaction in enumerate(interaction_types) if interaction == "like"]
    irrelevant_indices = [i for i, interaction in enumerate(interaction_types) if interaction == "dislike"]

    # Precision and Recall calculations
    relevant_predictions = [predictions[i] for i in relevant_indices]
    irrelevant_predictions = [predictions[i] for i in irrelevant_indices]

    precision = len(relevant_predictions) / len(predictions) if predictions else 0
    recall = len(relevant_predictions) / len(relevant_indices) if relevant_indices else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # RMSE and MAE for all ratings
    rmse = sqrt(mean_squared_error(true_ratings, predictions))
    mae = mean_absolute_error(true_ratings, predictions)

    # Accuracy based on exact matches for "like" or "dislike"
    accuracy = sum(
        (interaction == "like" and pred >= 0) or (interaction == "dislike" and pred < 0)
        for pred, interaction in zip(predictions, interaction_types)
    ) / len(predictions) if predictions else 0

    return {
        "rmse": rmse,
        "mae": mae,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }

def train_interactive_model(engine, retrain=False):
    """
    Incrementally train or retrain an interactive filtering model using new interaction data.

    Args:
        engine: SQLAlchemy engine for database connection.
        retrain (bool): If True, retrain the model from scratch. Defaults to False.

    Returns:
        dict: Contains the trained model, metadata, and training message.
    """
    try:
        user_id = st.session_state.get("user_id")  # Fetch user_id from session state

        if not user_id:
            logging.error("User ID not found in session state.")
            return {"message": "User ID is missing. Ensure the session state is properly set."}

        # Initialize last_training_time
        last_training_time = None
        
        # Check if retrain is requested
        if retrain:
            logging.info("Retrain flag is set. Attempting to fetch the existing model for retraining.")
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    with open(metadata_path, "rb") as f:
                        model_metadata = pickle.load(f)
                    logging.info("Existing model and metadata loaded successfully for retraining.")
                except (pickle.UnpicklingError, FileNotFoundError, KeyError) as e:
                    logging.warning(f"Failed to load existing model or metadata: {e}. Training from scratch.")
                    model = SVDpp()
            else:
                logging.info("No existing model found. Training from scratch.")
                model = SVDpp()
        else:
            # Load model incrementally if available
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                try:
                    logging.info("Loading existing model for incremental training.")
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    with open(metadata_path, "rb") as f:
                        model_metadata = pickle.load(f)
                    last_training_time = model_metadata.get("last_training_time")
                    logging.info(f"Model last trained on: {last_training_time}")
                except (pickle.UnpicklingError, FileNotFoundError, KeyError) as e:
                    logging.warning(f"Failed to load model or metadata: {e}. Training from scratch.")
                    model = SVDpp()
                    last_training_time = None
            else:
                logging.info("No existing model found. Training from scratch.")
                model = SVDpp()
                last_training_time = None

        # Fetch new interaction data since the last training
        query = text("""
            SELECT user_id, imdb_id, COALESCE(rating, 0) AS rating, interaction_type, timestamp
            FROM user_interactions
            WHERE rating IS NOT NULL AND user_id = :user_id
        """ + (" AND timestamp > :last_training_time" if last_training_time else ""))

        params = {"user_id": user_id}
        if last_training_time:
            params["last_training_time"] = last_training_time

        with engine.connect() as connection:
            result = connection.execute(query, params)
            data = pd.DataFrame(result.fetchall(), columns=result.keys())

        if data.empty:
            logging.info("No new interaction data found for training.")
            return {"message": "No new data available for training."}

        # Train the model
        logging.info(f"Preparing interaction data with {len(data)} records for training.")
        reader = Reader(rating_scale=(0, 10))
        dataset = Dataset.load_from_df(data[["user_id", "imdb_id", "rating"]], reader)
        trainset = dataset.build_full_trainset()
        model.fit(trainset)

        # Save updated model and metadata
        logging.info(f"Saving the updated model to {model_path}...")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logging.info(f"Saving updated model metadata to {metadata_path}...")
        model_metadata = {
            "last_training_time": pd.Timestamp.now().isoformat(),
            "trained_on": len(data),
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(model_metadata, f)

        # Evaluate the model using predictions
        predictions = [model.predict(uid, iid).est for uid, iid in zip(data["user_id"], data["imdb_id"])]
        interaction_types = data["interaction_type"].tolist()
        metrics = evaluate_model(predictions, data["rating"], interaction_types)
        engagement_rate = len(data) / len(data["imdb_id"].unique()) if len(data["imdb_id"].unique()) > 0 else None

        # Log metrics to the database
        log_model_metrics(engine, {
            "user_id": user_id,
            "num_movies": len(data["imdb_id"].unique()),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "coverage": len(set(data["imdb_id"])) / len(data["imdb_id"].unique()),
            "engagement_rate": engagement_rate,
            "notes": "Model retrained with updated interaction data."
        })

        logging.info("Interactive filtering model updated successfully.")
        return {
            "model": model,
            "metadata": model_metadata,
            "message": "Interactive model updated successfully!"
        }

    except Exception as e:
        logging.error(f"Error training interactive filtering model: {e}")
        return {"message": "Error occurred during model training."}

# def get_top_recommendations(engine, user_id, top_n=5):
#     """
#     Generate top-N movie recommendations for a user using the trained interactive filtering model.

#     Args:
#         engine: SQLAlchemy engine for database connection.
#         user_id (int): The ID of the user.
#         top_n (int): Number of recommendations to generate.

#     Returns:
#         list of dict: List of recommended movies with metadata and predicted ratings.
#     """
#     model_path = "interactive_model.pkl"
#     if not os.path.exists(model_path):
#         logging.error("Interactive filtering model not found. Train the model first.")
#         return [{"message": "Interactive filtering model not found."}]

#     # Load the trained interactive filtering model
#     try:
#         with open(model_path, "rb") as f:
#             model = pickle.load(f)
#     except Exception as e:
#         logging.error(f"Error loading interactive filtering model: {e}")
#         return [{"message": "Error loading interactive filtering model."}]

#     try:
#         with engine.connect() as connection:
#             # Fetch all IMDb IDs from the movies table
#             movie_query = text("SELECT DISTINCT imdb_id FROM movies")
#             movie_ids = [row["imdb_id"] for row in connection.execute(movie_query)]

#             # Fetch IMDb IDs already interacted with by the user
#             interacted_query = text("""
#                 SELECT movies.imdb_id
#                 FROM user_interactions
#                 JOIN movies ON user_interactions.movie_id = movies.movie_id
#                 WHERE user_interactions.user_id = :user_id
#             """)
#             interacted_movies = [row["imdb_id"] for row in connection.execute(interacted_query, {"user_id": user_id})]

#             recommendations = []

#             # Predict ratings for movies the user hasn't interacted with
#             for imdb_id in movie_ids:
#                 if imdb_id not in interacted_movies:
#                     predicted_rating = model.predict(user_id, imdb_id).est
#                     recommendations.append((imdb_id, predicted_rating))

#             # Sort by predicted rating in descending order
#             recommendations.sort(key=lambda x: x[1], reverse=True)

#             # Fetch metadata for the top-N recommendations
#             top_recommendations = recommendations[:top_n]
#             movie_metadata_query = text("""
#                 SELECT imdb_id, title, genre, overview, director
#                 FROM movies
#                 WHERE imdb_id IN :movie_ids
#             """)
#             metadata_results = connection.execute(
#                 movie_metadata_query, {"movie_ids": tuple(rec[0] for rec in top_recommendations)}
#             ).fetchall()

#             metadata_map = {
#                 row["imdb_id"]: {
#                     "title": row["title"],
#                     "genre": row["genre"],
#                     "overview": row["overview"],
#                     "director": row["director"],
#                 }
#                 for row in metadata_results
#             }

#             # Construct final output
#             return [
#                 {
#                     "imdb_id": rec[0],
#                     "predicted_rating": rec[1],
#                     "title": metadata_map.get(rec[0], {}).get("title", "Unknown"),
#                     "genre": metadata_map.get(rec[0], {}).get("genre", "Unknown"),
#                     "overview": metadata_map.get(rec[0], {}).get("overview", "No overview available."),
#                     "director": metadata_map.get(rec[0], {}).get("director", "Unknown"),
#                 }
#                 for rec in top_recommendations
#             ]

#     except Exception as e:
#         logging.error(f"Error generating recommendations: {e}")
#         return [{"message": "Error generating recommendations."}]