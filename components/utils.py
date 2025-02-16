import os
import sys
import logging
import requests
import pandas as pd
from time import sleep
import streamlit as st
from datetime import datetime
from sqlalchemy.sql import text
from argon2 import PasswordHasher
from sqlalchemy import create_engine

# Dynamically add paths to `sys.path` for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))  # Project root
components_path = os.path.join(project_root, "components")
config_path = os.path.join(project_root, "config")
data_fetcher_path = os.path.join(project_root, "data_fetcher")

# Add these paths to sys.path if necessary
sys.path.insert(0, components_path)
sys.path.insert(0, config_path)
sys.path.insert(0, data_fetcher_path)

# Database configuration
from config.config import DATABASE, TMDB_API_KEY

# TMDb API
BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", TMDB_API_KEY)

# Database Connection
@st.experimental_singleton
def get_sqlalchemy_engine():
    """
    Establish and return a SQLAlchemy engine for database connection.
    """
    try:
        DATABASE_URI = f"postgresql+psycopg2://{DATABASE['USER']}:{DATABASE['PASSWORD']}@{DATABASE['HOST']}:{DATABASE['PORT']}/{DATABASE['DB_NAME']}"
        logging.info(f"Connecting to database using URI: {DATABASE_URI}")

        engine = create_engine(DATABASE_URI)

        with engine.connect() as connection:
            logging.info("Database connection established successfully.")
        return engine
    except KeyError as e:
        logging.error(f"Missing database configuration key: {e}")
        return None
    except ModuleNotFoundError as e:
        logging.error(f"Missing required library: {e}. Ensure `psycopg2` and `SQLAlchemy` are installed.")
        return None
    except Exception as e:
        logging.error(f"Failed to establish database connection: {e}")
        return None

# Initialize Password Hasher
ph = PasswordHasher()

def get_movie_id_by_title(movie_title):
    """
    Fetch the movie_id for a given movie title.

    Args:
        movie_title (str): The title of the movie.

    Returns:
        int: The movie_id, or None if not found.
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logging.error("Failed to establish database connection.")
        return None
    try:
        query = text("SELECT movie_id FROM movies WHERE title = :movie_title")
        with engine.connect() as conn:
            result = conn.execute(query, {"movie_title": movie_title}).fetchone()
        return result["movie_id"] if result else None
    except Exception as e:
        logging.error(f"Failed to fetch movie_id for title '{movie_title}': {e}")
        return None

# TMDB Actor Search
def search_tmdb_actors(query, engine, retries=3, delay=2):
    """
    Fetch actors matching the query from TMDB and filter them based on the movies table.

    Args:
        query (str): Actor name or partial name to search for.
        engine: SQLAlchemy database engine.
        retries (int): Number of retry attempts for the API call.
        delay (int): Delay between retries.

    Returns:
        list: List of actor names relevant to the database movies.
    """
    search_url = f"{BASE_URL}/search/person?api_key={TMDB_API_KEY}&query={query}"
    for attempt in range(retries):
        try:
            # Fetch results from the TMDB API
            response = requests.get(search_url)
            if response.status_code == 429:  # Too many requests
                logging.warning(f"Rate limit reached. Retrying in {delay} seconds...")
                sleep(delay)
                continue
            response.raise_for_status()

            results = response.json().get("results", [])
            if not results:
                logging.info(f"No actors found for query: {query}")
                return []

            # Fetch movie titles from the database
            with engine.connect() as conn:
                query = text("SELECT title FROM movies WHERE title IS NOT NULL")
                db_movies = {row[0] for row in conn.execute(query).fetchall()}  # Use a set for faster lookups

            # Filter actors based on database movies
            actor_names = [
                actor["name"]
                for actor in results
                if any(
                    movie.get("title") in db_movies for movie in actor.get("known_for", [])
                )
            ]

            return actor_names
        except requests.exceptions.RequestException as e:
            logging.error(f"Error searching for actors: {e}")
        sleep(delay)
    return []

# Validate User Credentials
def validate_user(username, password):
    """
    Validate user credentials by checking the password against the hashed password in the database.

    Args:
        username (str): Username provided by the user.
        password (str): Plain-text password provided by the user.

    Returns:
        bool: True if the credentials are valid, False otherwise.
    """
    try:
        engine = get_sqlalchemy_engine()
        if not engine:
            logging.error("Database connection failed during user validation.")
            return False

        with engine.connect() as connection:
            query = text("SELECT password FROM users WHERE username = :username")
            result = connection.execute(query, {"username": username}).fetchone()

        if result:
            stored_hashed_password = result[0]
            # Validate the password
            if ph.verify(stored_hashed_password, password):
                logging.info(f"User '{username}' successfully validated.")
                return True
            else:
                logging.warning(f"Validation failed for user '{username}'. Incorrect password.")
        else:
            logging.warning(f"User '{username}' not found.")
        return False
    except Exception as e:
        logging.error(f"Error validating user '{username}': {e}")
        return False

# Fetch User Info
def fetch_user_info(username):
    """
    Fetch the basic information of a user from the database.

    Args:
        username (str): Username of the user.

    Returns:
        dict: User details including username and email if found, None otherwise.
    """
    try:
        engine = get_sqlalchemy_engine()
        if not engine:
            logging.error("Database connection failed during user info fetch.")
            return None

        with engine.connect() as conn:
            query = text("SELECT username, email FROM users WHERE username = :username")
            result = conn.execute(query, {"username": username}).fetchone()

        if result:
            logging.info(f"User info fetched successfully for '{username}'.")
            return {"username": result[0], "email": result[1]}
        else:
            logging.warning(f"No user found with username '{username}'.")
            return None
    except Exception as e:
        logging.error(f"Error fetching user info for '{username}': {e}")
        return None

# Register User
def register_user(name, email, username, password):
    """
    Register a new user by hashing their password and storing it in the database.

    Args:
        name (str): Full name.
        email (str): Email address.
        username (str): Username.
        password (str): Plain-text password.

    Returns:
        bool: True if registration is successful, False otherwise.
    """
    try:
        engine = get_sqlalchemy_engine()
        if not engine:
            logging.error("Database connection failed during user registration.")
            return False

        hashed_password = ph.hash(password)
        query = text("""
            INSERT INTO users (name, email, username, password)
            VALUES (:name, :email, :username, :password)
        """)
        with engine.connect() as connection:
            connection.execute(
                query,
                {
                    "name": name,
                    "email": email,
                    "username": username,
                    "password": hashed_password
                }
            )
        logging.info(f"User '{username}' registered successfully.")
        return True
    except Exception as e:
        logging.error(f"Error registering user '{username}': {e}")
        return False

# Check Email Exists
def check_email_exists(email):
    """
    Check if an email exists in the database.

    Args:
        email (str): Email address to check.

    Returns:
        bool: True if the email exists, False otherwise.
    """
    try:
        engine = get_sqlalchemy_engine()
        if not engine:
            logging.error("Database connection failed while checking email existence.")
            return False

        with engine.connect() as connection:
            query = text("SELECT 1 FROM users WHERE email = :email")
            result = connection.execute(query, {"email": email}).fetchone()

        if result:
            logging.info(f"Email '{email}' exists in the database.")
        else:
            logging.warning(f"Email '{email}' does not exist in the database.")

        return bool(result)
    except Exception as e:
        logging.error(f"Error checking email existence for '{email}': {e}")
        return False

# Reset User Password
def reset_password(email, new_password):
    """
    Reset a user's password.

    Args:
        email (str): User's email.
        new_password (str): New password.

    Returns:
        bool: True if reset is successful, False otherwise.
    """
    try:
        engine = get_sqlalchemy_engine()
        if not engine:
            logging.error("Database connection failed during password reset.")
            return False

        hashed_password = ph.hash(new_password)
        query = text("UPDATE users SET password = :password WHERE email = :email")
        with engine.connect() as connection:
            connection.execute(query, {"password": hashed_password, "email": email})
        logging.info(f"Password reset successfully for email: {email}")
        return True
    except Exception as e:
        logging.error(f"Error resetting password for email '{email}': {e}")
        return False
    
# Additional Functionality
def log_model_metrics(engine, metrics_data):
    """
    Logs model performance metrics into the model_metrics table in the database.

    Args:
        engine: SQLAlchemy engine for database connection.
        metrics_data: Dictionary containing all relevant metric values.
    """
    query = text("""
        INSERT INTO model_metrics (
            user_id, num_movies, timestamp, precision, recall, f1_score, accuracy, rmse, mae, coverage, 
            engagement_rate, notes
        ) VALUES (
            :user_id, :num_movies, :timestamp, :precision, :recall, :f1_score, :accuracy, :rmse, :mae, :coverage, 
            :engagement_rate, :notes
        )
        ON CONFLICT (user_id) DO UPDATE 
        SET 
            num_movies = EXCLUDED.num_movies,
            timestamp = EXCLUDED.timestamp,
            precision = EXCLUDED.precision,
            recall = EXCLUDED.recall,
            f1_score = EXCLUDED.f1_score,
            accuracy = EXCLUDED.accuracy,
            rmse = EXCLUDED.rmse,
            mae = EXCLUDED.mae,
            coverage = EXCLUDED.coverage,
            engagement_rate = EXCLUDED.engagement_rate,
            notes = EXCLUDED.notes
    """)
    
    # Ensure timestamp is current if not provided
    metrics_data['timestamp'] = metrics_data.get('timestamp', datetime.utcnow())
    
    try:
        with engine.connect() as conn:
            conn.execute(query, metrics_data)
        logging.info("Model metrics logged successfully.")
    except Exception as e:
        logging.error(f"Error logging model metrics: {e}")

def fetch_model_metrics(engine, user_id):
    """
    Fetch model performance metrics (e.g., RMSE, MAE, and other metrics) for a specific user from the database.

    Args:
        engine: SQLAlchemy engine for database connection.
        user_id: ID of the user for whom metrics are fetched.

    Returns:
        pd.DataFrame: DataFrame containing timestamp, RMSE, MAE, and additional metrics.
    """
    query = text("""
        SELECT 
            timestamp,
            num_movies,
            precision, 
            recall, 
            f1_score, 
            accuracy, 
            rmse, 
            mae, 
            coverage, 
            engagement_rate,
            notes
        FROM model_metrics
        WHERE user_id = :user_id
        ORDER BY timestamp DESC
    """)
    try:
        with engine.connect() as conn:
            metrics = pd.read_sql_query(query, conn, params={"user_id": user_id})
        return metrics
    except Exception as e:
        logging.error(f"Error fetching model metrics for user_id {user_id}: {e}")
        return pd.DataFrame()
    
# Fetch Unique Genres
def fetch_unique_genres():
    """
    Fetch unique genres from the movies table.

    Returns:
        list: A list of unique genres sorted alphabetically.
    """
    try:
        engine = get_sqlalchemy_engine()
        query = text("""
            SELECT DISTINCT UNNEST(STRING_TO_ARRAY(genre, ', ')) AS genre
            FROM movies;
        """)
        with engine.connect() as conn:
            genres = [row[0] for row in conn.execute(query).fetchall()]
        return sorted(set(genres))
    except Exception as e:
        st.error("Error fetching genres.")
        logging.error(f"Error in fetch_unique_genres: {e}")
        return []

# Fetch Unique Directors
def fetch_unique_directors():
    """
    Fetch unique directors from the movies table.

    Returns:
        list: A list of unique director names sorted alphabetically.
    """
    try:
        engine = get_sqlalchemy_engine()
        query = text("SELECT DISTINCT director FROM movies WHERE director IS NOT NULL;")
        with engine.connect() as conn:
            directors = [row[0] for row in conn.execute(query).fetchall()]
        return sorted(directors)
    except Exception as e:
        logging.error(f"Error in fetch_unique_directors: {e}")
        return []