import os
import sys
import json
import spacy
import pickle
import logging
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add relevant folders to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
components_path = os.path.join(project_root, "components")

sys.path.insert(0, components_path)

from components.utils import get_sqlalchemy_engine
from interaction_handler_streamlit import train_interactive_model  # Import training logic

# Define log file path
log_file_path = os.path.join(os.getcwd(), "narrative_recommender.log")

# Create a named logger
logger = logging.getLogger("narrative_recommender")
logger.setLevel(logging.DEBUG)  # Set the logger level

# File handler for logging to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)

BASE_DIR = "recommendation"
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "scene_embeddings.npy")
METADATA_FILE = os.path.join(BASE_DIR, "scene_metadata.npy")
INTER_MODEL_FILE = "interactive_model.pkl"
INTER_METADATA_FILE = "interactive_model_metadata.pkl"
SYNONYM_FILE_PATH = os.path.join(BASE_DIR, "synonym_enrichment.json")

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Fetch User Interactions
def fetch_user_interactions(engine, user_id):
    """
    Fetch IMDb IDs of movies interacted with by a user.

    Args:
        engine: SQLAlchemy engine for database connection.
        user_id (int): User ID.

    Returns:
        list: List of IMDb IDs for movies interacted with by the user.
    """
    try:
        query = text("""
            SELECT DISTINCT movies.imdb_id
            FROM user_interactions
            JOIN movies ON user_interactions.movie_id = movies.movie_id
            WHERE user_interactions.user_id = :user_id
        """)
        with engine.connect() as connection:
            result = connection.execute(query, {"user_id": user_id}).fetchall()
        return [row["imdb_id"] for row in result]
    except Exception as e:
        logger.error(f"Error fetching user interactions for user_id {user_id}: {e}")
        return []

# Preprocess Scene Data
def preprocess_scene_data(scene_data):
    """
    Preprocess and clean scene data to align with expected formats and concatenate fields.

    Args:
        scene_data (pd.DataFrame): Raw data fetched from the database.

    Returns:
        pd.Series: Concatenated and cleaned textual data for embeddings.
    """
    try:
        # Validate required columns
        required_columns = {"action_summary", "tone", "key_elements", "thematic_analysis"}
        missing_columns = required_columns - set(scene_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Fill missing values for text-based columns
        for column in ["action_summary", "thematic_analysis"]:
            scene_data[column] = scene_data[column].fillna("Missing").astype(str)

        # Validate and clean tone and key_elements fields
        def validate_json_array(value):
            try:
                if isinstance(value, str):
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return value
                if isinstance(value, list):
                    return json.dumps(value)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON detected, defaulting to empty array: {value}")
            return json.dumps([])

        scene_data["tone"] = scene_data["tone"].apply(validate_json_array)
        scene_data["key_elements"] = scene_data["key_elements"].apply(validate_json_array)

        # Concatenate fields into a single textual representation
        scene_texts = (
            scene_data["action_summary"] * 3 + " " +
            scene_data["tone"] * 2 + " " +
            scene_data["key_elements"] * 2 + " " +
            scene_data["thematic_analysis"] * 3
        )

        logger.info("Scene data preprocessing and concatenation completed successfully.")
        return scene_texts.drop_duplicates()
    except Exception as e:
        logger.error(f"Error during preprocessing scene data: {e}")
        return pd.Series(dtype="object")

# Fetch Scene Data
def fetch_scene_data(engine):
    """
    Fetch and preprocess scene data from the nine_act_annotations table.

    Args:
        engine: SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: Cleaned and validated scene data.
    """
    try:
        query = text("""
            SELECT imdb_id, chunk_number, action_summary, tone, key_elements, thematic_analysis
            FROM nine_act_annotations
        """)
        with engine.connect() as connection:
            scene_data = pd.read_sql_query(query, connection)
            
        return scene_data
    except Exception as e:
        logger.error(f"Error fetching scene data: {e}")
        return pd.DataFrame()

# Generate Scene Embeddings
def generate_scene_embeddings(preprocessed_scene_texts, scene_data, model, batch_size=500):
    """
    Generate and save embeddings for scene data.

    Args:
        scene_data (pd.DataFrame): DataFrame with scene details.
        model: SentenceTransformer model for embedding generation.
        batch_size (int): Number of samples to process in one batch.

    Returns:
        np.ndarray: Embeddings array.
    """
    try:
        if preprocessed_scene_texts.empty:
            raise ValueError("Preprocessed scene texts are empty.")

        embeddings = []
        logger.info("Starting batch embedding generation...")
        for start in range(0, len(preprocessed_scene_texts), batch_size):
            batch_texts = preprocessed_scene_texts[start:start + batch_size].tolist()
            logger.info(f"Processing batch {start // batch_size + 1} with {len(batch_texts)} items...")
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
            embeddings.append(batch_embeddings)

        # Combine into a single NumPy array
        embeddings = np.vstack(embeddings)

        # Validate dimensions
        if embeddings.ndim != 2:
            logger.error(f"Inconsistent embedding dimensions: Expected 2D, got {embeddings.ndim}D")
            raise ValueError("Generated embeddings are not 2D. Ensure consistent dimensions.")

        # Save embeddings and metadata
        np.save(EMBEDDINGS_FILE, embeddings)
        logger.info(f"Embeddings saved to {EMBEDDINGS_FILE}")

        metadata = scene_data["imdb_id"].to_list()
        np.save(METADATA_FILE, metadata)
        logger.info(f"Metadata saved to {METADATA_FILE}")

        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

# Load Precomputed Embeddings
def load_embeddings():
    """
    Load precomputed embeddings and metadata from files.

    Returns:
        tuple: (embeddings, metadata)
    """
    try:
        embeddings = np.load(EMBEDDINGS_FILE)
        metadata = np.load(METADATA_FILE, allow_pickle=True)
        logger.info("Loaded embeddings and metadata from files.")
        logger.debug(f"Embeddings length: {len(embeddings)}, Metadata length: {len(metadata)}")

        # Align embeddings and metadata
        if len(metadata) > len(embeddings):
            logger.warning("Truncating metadata to match embeddings length.")
            metadata = metadata[:len(embeddings)]
        elif len(embeddings) > len(metadata):
            logger.warning("Truncating embeddings to match metadata length.")
            embeddings = embeddings[:len(metadata)]
        # Validate embeddings
        if embeddings.ndim != 2:
            raise ValueError(f"Loaded embeddings have invalid dimensions: {embeddings.shape}")

        return embeddings, metadata
    except FileNotFoundError:
        logger.error("Embedding or metadata files not found. Ensure they are precomputed.")
        raise
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise

def preprocess_user_input(user_scene):
    """
    Preprocess user input to improve alignment with embedding and metadata.

    Args:
        user_scene (str): Raw user input describing a scene or theme.

    Returns:
        str: Preprocessed and enhanced user input.
    """
    try:
        # Load synonyms from the JSON file directly
        try:
            with open(SYNONYM_FILE_PATH, "r") as f:
                synonym_enrichment = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load synonyms: {e}")
            synonym_enrichment = {}

        # Process the user input with spaCy
        doc = nlp(user_scene.lower())  # Lowercase and tokenize in one step

        # Tokenize, remove stop words, and lemmatize using spaCy
        tokens = {token.lemma_ for token in doc if not token.is_stop and token.is_alpha}

        # Enrich input with synonyms
        enriched_tokens = set()
        for token in tokens:
            enriched_tokens.add(token)
            if token in synonym_enrichment:
                enriched_tokens.update(synonym_enrichment[token])

        return enriched_tokens
    except Exception as e:
        logging.error(f"Error during user input preprocessing: {e}")
        return user_scene

# Find Similar Scenes
def find_similar_scenes(user_scene, embeddings, metadata, model, top_n=5):
    """
    Find scenes similar to the user's input.

    Args:
        user_scene (str): User-provided scene description.
        embeddings (np.ndarray): Precomputed scene embeddings.
        metadata (list): Metadata corresponding to the embeddings.
        model: SentenceTransformer model for embedding generation.
        top_n (int): Number of similar scenes to return.

    Returns:
        list of dict: List of recommended scenes and their metadata.
    """
    try:
        # Generate embedding for the user-provided scene
        logger.info("Generating embedding for user-provided scene...")
        user_embedding = model.encode([user_scene], convert_to_tensor=False)

        # Validate embedding dimensions
        if user_embedding.ndim == 1:
            user_embedding = user_embedding.reshape(1, -1)
        elif user_embedding.ndim > 2:
            raise ValueError(f"User embedding has invalid dimensions: {user_embedding.shape}")

        # Compute cosine similarity
        logger.info("Computing cosine similarity...")
        similarities = cosine_similarity(user_embedding, embeddings)[0]

        # Sort by similarity and fetch top N
        top_indices = similarities.argsort()[-top_n:][::-1]
        seen_titles = set()  # Keep track of seen titles
        top_scenes = []

        for i in top_indices:
            if metadata[i] not in seen_titles:
                top_scenes.append({"imdb_id": metadata[i], "similarity": similarities[i]})
                seen_titles.add(metadata[i])

        logger.info(f"Top {top_n} similar scenes: {top_scenes}")
        return top_scenes
    except Exception as e:
        logger.error(f"Error finding similar scenes: {e}")
        return []

# Generate Narrative Recommendations
def generate_narrative_recommendations(engine, user_scene, top_n=5):
    try:
        # Load embeddings and metadata
        embeddings, metadata = load_embeddings()

        if not user_scene.strip():
            # Fallback to popular or trending movies if user_scene is empty
            logger.info("User scene input is empty. Fetching fallback recommendations.")
            fallback_query = text("""
                SELECT imdb_id, title, genre, overview, director
                FROM movies
                ORDER BY popularity DESC, vote_count DESC
                LIMIT :top_n
            """)
            with engine.connect() as connection:
                fallback_movies = connection.execute(fallback_query, {"top_n": top_n}).fetchall()
            return [
                {
                    "imdb_id": row["imdb_id"],
                    "title": row["title"],
                    "genre": row["genre"],
                    "overview": row["overview"],
                    "director": row["director"],
                    "score": None,  # No similarity score for fallback
                }
                for row in fallback_movies
            ]

        # Generate content-based scores
        model = SentenceTransformer("all-MiniLM-L12-v2")
        user_embedding = model.encode([user_scene], convert_to_tensor=False).reshape(1, -1)
        cosine_scores = cosine_similarity(user_embedding, embeddings)[0]

        # Sort by similarity and fetch top N
        top_indices = cosine_scores.argsort()[-top_n:][::-1]
        seen_titles = set()
        recommendations = [
            {"imdb_id": metadata[i], "similarity": cosine_scores[i]}
            for i in top_indices if metadata[i] not in seen_titles and not seen_titles.add(metadata[i])
        ]

        # Fetch metadata for recommendations
        imdb_ids = [rec["imdb_id"] for rec in recommendations]
        metadata_query = text("""
            SELECT imdb_id, title, genre, overview, director
            FROM movies
            WHERE imdb_id = ANY(:imdb_ids)
        """)
        with engine.connect() as connection:
            movie_metadata = connection.execute(metadata_query, {"imdb_ids": imdb_ids}).fetchall()

        metadata_map = {row["imdb_id"]: dict(row) for row in movie_metadata}
        for rec in recommendations:
            metadata_row = metadata_map.get(rec["imdb_id"], {})
            rec.update({
                "title": metadata_row.get("title", "Unknown"),
                "genre": metadata_row.get("genre", "Unknown"),
                "overview": metadata_row.get("overview", "No overview available."),
                "director": metadata_row.get("director", "Unknown"),
            })

        return recommendations
    except Exception as e:
        logger.error(f"Error generating narrative recommendations: {e}")
        return [{"message": "An error occurred while generating narrative recommendations."}]

# Generate Hybrid Recommendations
def generate_hybrid_recommendations(engine, user_scene, user_id, content_weight=1.0, collab_weight=0, top_n=5):
    try:
        # Load narrative embeddings and metadata
        embeddings, metadata = load_embeddings()
        metadata_list = metadata.tolist()  # Convert NumPy array to list
        assert len(embeddings) == len(metadata_list), "Embeddings and metadata lengths do not match"

        # Fetch user preferences from the database
        preferences_query = text("""
            SELECT preferred_genres, preferred_styles, favorite_movies, preferred_directors
            FROM users
            WHERE user_id = :user_id
        """)
        with engine.connect() as connection:
            user_preferences = connection.execute(preferences_query, {"user_id": user_id}).fetchone()

        # Check if the interactive model exists
        if not os.path.exists(INTER_MODEL_FILE):
            logger.warning("Interactive filtering model not found. Training the model...")
            training_result = train_interactive_model(engine)
            if not training_result or training_result.get("message") != "Interactive model trained successfully!":
                return [{"message": "Failed to train interactive filtering model."}]

        with open(INTER_MODEL_FILE, "rb") as f:
            interactive_model = pickle.load(f)

        # Fetch metadata for all IMDb IDs
        metadata_query = text("""
            SELECT imdb_id, title, genre, keywords, director
            FROM movies
            WHERE imdb_id = ANY(:imdb_ids)
        """)
        with engine.connect() as connection:
            movie_metadata_results = connection.execute(metadata_query, {"imdb_ids": metadata_list}).fetchall()

        # Convert results to a dictionary for easy lookup
        movie_metadata = {row["imdb_id"]: dict(row) for row in movie_metadata_results}

        # Generate content-based scores
        model = SentenceTransformer("all-MiniLM-L12-v2")
        user_embedding = model.encode([user_scene], convert_to_tensor=False).reshape(1, -1)
        cosine_scores = cosine_similarity(user_embedding, embeddings)[0]
        assert len(cosine_scores) == len(metadata_list), "Cosine scores and metadata lengths do not match"

        def adjust_content_score(movie, user_preferences):
            score = 1.0
            if user_preferences["preferred_genres"] and any(
                genre in user_preferences["preferred_genres"].split(",") for genre in movie["genre"].split(",")
            ):
                score *= 1.5
            if user_preferences["preferred_styles"] and any(
                style in user_preferences["preferred_styles"].split(",") for style in movie.get("keywords", "").split(",")
            ):
                score *= 1.3
            if user_preferences["preferred_directors"] and movie["director"] in user_preferences["preferred_directors"].split(","):
                score *= 1.6
            if user_preferences["favorite_movies"] and movie["title"] in user_preferences["favorite_movies"].split(","):
                score *= 2.0
            return score

        content_scores = {
            imdb_id: cosine_scores[i] * adjust_content_score(movie_metadata.get(imdb_id, {}), user_preferences)
            for i, imdb_id in enumerate(metadata_list)
        }

        # Generate collaborative scores
        movie_query = text("SELECT DISTINCT imdb_id FROM movies")
        with engine.connect() as connection:
            movie_ids = [row[0] for row in connection.execute(movie_query)]

        interactive_scores = {
            movie_id: interactive_model.predict(user_id, movie_id).est for movie_id in movie_ids
        }

        # Combine content and collaborative scores
        if content_weight == 1.0 and collab_weight == 0.0:
            hybrid_scores = content_scores
        elif content_weight == 0.0 and collab_weight == 1.0:
            hybrid_scores = interactive_scores
        else:
            hybrid_scores = {
                imdb_id: content_weight * content_scores.get(imdb_id, 0) + collab_weight * interactive_scores.get(imdb_id, 0)
                for imdb_id in set(content_scores.keys()).union(interactive_scores.keys())
            }

        # Filter and sort scores
        seen_titles = set()
        filtered_scores = {
            imdb_id: score for imdb_id, score in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            if imdb_id not in seen_titles and not seen_titles.add(imdb_id)
        }

        # Fetch metadata for top recommendations
        top_ids = list(filtered_scores.keys())[:top_n]
        metadata_query = text("""
            SELECT imdb_id, title, genre, overview, director
            FROM movies
            WHERE imdb_id = ANY(:top_ids)
        """)
        with engine.connect() as connection:
            metadata_results = connection.execute(metadata_query, {"top_ids": top_ids}).fetchall()

        recommendations = [
            {
                "imdb_id": row["imdb_id"],
                "title": row["title"],
                "genre": row["genre"],
                "overview": row["overview"],
                "director": row["director"],
                "score": filtered_scores[row["imdb_id"]],
                "similarity": cosine_scores[metadata_list.index(row["imdb_id"])] if row["imdb_id"] in metadata_list else 0,
            }
            for row in metadata_results
        ]

        return recommendations
    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {e}")
        return [{"message": "An error occurred while generating hybrid recommendations."}]

def metadata_evaluation(user_scene, recommendations, movie_metadata):
    """
    Evaluate recommendations using metadata overlap.

    Args:
        user_scene (str): User-provided scene description.
        recommendations (list): List of recommended movie IMDb IDs and metadata.
        movie_metadata (dict): Dictionary mapping IMDb IDs to metadata.

    Returns:
        dict: Metrics including precision, recall, and F1-score.
    """
    try:
        # Load synonyms from the JSON file directly
        try:
            with open(SYNONYM_FILE_PATH, "r") as f:
                synonym_enrichment = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load synonyms: {e}")
            synonym_enrichment = {}

        # Preprocess user input
        user_keywords = preprocess_user_input(user_scene)

        relevant_count = 0
        for rec in recommendations:
            imdb_id = rec["imdb_id"]
            # Preprocess metadata directly within the loop
            metadata_keywords = preprocess_user_input(movie_metadata.get(imdb_id, ""))
            if user_keywords & metadata_keywords:  # Check intersection
                relevant_count += 1

        precision = relevant_count / len(recommendations) if recommendations else 0
        recall = relevant_count / len(user_keywords) if user_keywords else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {"precision": precision, "recall": recall, "f1_score": f1_score}
        return metrics
    except Exception as e:
        logging.error(f"Error in metadata evaluation: {e}")
        return {"precision": 0, "recall": 0, "f1_score": 0}

def similarity_analysis(recommendations, similarity_threshold=0.6):
    """
    Analyze cosine similarity scores of recommendations.

    Args:
        recommendations (list): List of recommended movies with similarity scores.
        similarity_threshold (float): Threshold to consider a recommendation as relevant.

    Returns:
        dict: Metrics including mean similarity and precision above threshold.
    """
    scores = [rec["similarity"] for rec in recommendations if "similarity" in rec]

    mean_similarity = np.mean(scores) if scores else 0
    above_threshold_count = len([score for score in scores if score >= similarity_threshold])
    precision = above_threshold_count / len(scores) if scores else 0
    
    # Recall: Proportion of recommendations above the threshold out of all relevant recommendations
    total_relevant_count = len([score for score in scores if score > 0])  # Non-zero similarity scores
    recall = above_threshold_count / total_relevant_count if total_relevant_count else 0

    # F1-Score: Harmonic mean of precision and recall
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "recall": recall,
        "f1_score": f1_score,
        "mean_similarity": mean_similarity,
        "precision_above_threshold": precision,
        "above_threshold_count": above_threshold_count,
    }
    return metrics

def main():
    """
    Main function to preprocess data, generate, and save scene embeddings.
    """
    # Initialize SQLAlchemy engine
    engine = get_sqlalchemy_engine()

    # Fetch raw scene data from the database
    logger.info("Fetching scene data from database...")
    scene_data = fetch_scene_data(engine)

    if scene_data.empty:
        logger.error("No scene data found. Please check the database connection and content.")
        return

    # Preprocess scene data
    logger.info("Preprocessing scene data...")
    try:
        preprocessed_scene_texts = preprocess_scene_data(scene_data)
        if preprocessed_scene_texts.empty:
            logger.error("Preprocessed scene texts are empty. Exiting...")
            return
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    # Load the SentenceTransformer model
    logger.info("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L12-v2")

    # Generate and save embeddings
    logger.info("Generating scene embeddings...")
    try:
        embeddings = generate_scene_embeddings(preprocessed_scene_texts, scene_data, model)
        logger.info(f"Successfully generated and saved embeddings for {len(embeddings)} scenes.")
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        return
        
if __name__ == "__main__":
    main()