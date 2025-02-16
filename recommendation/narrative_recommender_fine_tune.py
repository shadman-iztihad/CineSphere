import os
import sys
import json
import random
import pickle
import logging
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
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
logger = logging.getLogger("narrative_recommender_streamlit")
logger.setLevel(logging.DEBUG)  # Set the logger level

# File handler for logging to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Clear existing handlers (if any) to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Add handlers to the logger
logger.addHandler(file_handler)

BASE_DIR = "recommendation"
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "scene_embeddings.npy")
METADATA_FILE = os.path.join(BASE_DIR, "scene_metadata.npy")
INTER_MODEL_FILE = "interactive_model.pkl"
INTER_METADATA_FILE = "interactive_model_metadata.pkl"

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
        # Validate input type
        if not isinstance(scene_data, pd.DataFrame):
            raise ValueError("Input scene_data must be a pandas DataFrame.")
        # Validate required columns
        required_columns = {"chunk_text", "action_summary", "tone", "key_elements", "thematic_analysis"}
        missing_columns = required_columns - set(scene_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Fill missing values for text-based columns
        for column in ["chunk_text", "action_summary", "thematic_analysis"]:
            scene_data[column] = scene_data[column].fillna("Missing").astype(str)

        # Validate and clean tone and key_elements fields
        def validate_json_array(value):
            """
            Validate and normalize JSON array-like values. Handles lists, dictionaries, and malformed JSON.

            Args:
                value: Input value to validate.

            Returns:
                str: JSON string representation of the normalized array.
            """
            try:
                if isinstance(value, str):
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return json.dumps(parsed)  # Return list as JSON string
                    elif isinstance(parsed, dict):
                        # Flatten dictionary values into a list
                        flattened = [str(item) for sublist in parsed.values() for item in sublist]
                        return json.dumps(flattened)
                elif isinstance(value, list):
                    return json.dumps(value)  # Convert list to JSON string
                elif isinstance(value, dict):
                    # Handle raw dictionary (not JSON string)
                    flattened = [str(item) for sublist in value.values() for item in sublist]
                    return json.dumps(flattened)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Invalid JSON detected ({value}): {e}. Defaulting to empty array.")
            return json.dumps([])

        # Apply validation and clean data
        scene_data["tone"] = scene_data["tone"].apply(validate_json_array)
        scene_data["key_elements"] = scene_data["key_elements"].apply(validate_json_array)

        # Ensure all rows have valid `tone` and `key_elements`
        scene_data = scene_data[scene_data["tone"].notnull() & scene_data["key_elements"].notnull()]
        if scene_data.empty:
            logger.error("All rows were invalid after validation. Exiting preprocessing.")
            return pd.Series(dtype="object")

        # Concatenate fields into a single textual representation
        def safe_concatenate(value):
            """
            Safely processes and flattens JSON-like values for concatenation.

            Args:
                value: Input value (str, list, or other types).

            Returns:
                str: Flattened and concatenated string.
            """
            try:
                if isinstance(value, str):
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        # Extract relevant details from dictionaries if present
                        return " ".join(
                            item.get("description", item.get("prop", item.get("symbolism", str(item))))
                            if isinstance(item, dict) else str(item)
                            for item in parsed
                        )
                elif isinstance(value, list):
                    # Handle raw list of dictionaries
                    return " ".join(
                        item.get("description", item.get("prop", item.get("symbolism", str(item))))
                        if isinstance(item, dict) else str(item)
                        for item in value
                    )
                elif isinstance(value, dict):
                    # Flatten and join dictionary values
                    return " ".join(str(v) for v in value.values())
                else:
                    return str(value)
            except Exception as e:
                logger.warning(f"Failed to process value during concatenation: {value}. Skipping.")
                return ""

        scene_texts = (
            scene_data["action_summary"] * 3 + " " +
            scene_data["tone"].apply(safe_concatenate) * 2 + " " +
            scene_data["key_elements"].apply(safe_concatenate) * 2 + " " +
            scene_data["thematic_analysis"] * 3
        )

        # Ensure no empty or invalid entries
        scene_texts = scene_texts.str.strip()
        if scene_texts.empty or scene_texts.str.len().max() == 0:
            logger.error("Preprocessed scene data is empty or invalid.")
            return pd.Series(dtype="object")

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
            SELECT imdb_id, chunk_number, chunk_text, action_summary, tone, key_elements, thematic_analysis
            FROM nine_act_annotations
        """)
        with engine.connect() as connection:
            scene_data = pd.read_sql_query(query, connection)

        logger.debug(f"Fetched scene data columns: {scene_data.columns}")
        logger.debug(f"Sample scene data: {scene_data.head()}")

        # Validate fetched data
        if scene_data.empty:
            logger.error("Fetched scene data is empty. Ensure the database contains valid entries.")
            return pd.DataFrame()

        # Preprocess the data
        scene_texts = preprocess_scene_data(scene_data)
        if scene_texts.empty:
            logger.error("Preprocessed scene data is empty. Exiting fetch operation.")
            return pd.DataFrame()

        # Add processed scene texts as a new column
        scene_data["processed_text"] = scene_texts

        logger.info(f"Fetched and preprocessed {len(scene_data)} scenes from the database.")
        return scene_data
    except Exception as e:
        logger.error(f"Error fetching scene data: {e}")
        return pd.DataFrame()

def preprocess_scene_data_neutral(scene_data, columns_to_clean):
    """
    Preprocess specified columns in the DataFrame to ensure consistent types (e.g., convert floats to strings).

    Args:
        scene_data (pd.DataFrame): DataFrame with scene details.
        columns_to_clean (list): List of column names to preprocess.

    Returns:
        pd.DataFrame: Cleaned and consistent DataFrame.
    """
    def preprocess_value(value):
        try:
            if isinstance(value, float):
                return str(value)  # Convert float to string
            elif isinstance(value, list):
                # Recursively process list elements
                return [preprocess_value(item) for item in value]
            elif isinstance(value, dict):
                # Recursively process dictionary values
                return {k: preprocess_value(v) for k, v in value.items()}
            elif isinstance(value, (str, int)):
                return value  # Keep strings and integers as is
            else:
                return str(value)  # Convert other types to string
        except Exception as e:
            logger.warning(f"Error preprocessing value {value}: {e}")
            return "Invalid"

    # Process only specified columns
    for column in columns_to_clean:
        if column in scene_data.columns:
            scene_data[column] = scene_data[column].apply(preprocess_value)

    return scene_data

def split_training_data(training_data, chunk_size):
    """
    Split training data into smaller chunks.

    Args:
        training_data (list): List of InputExample objects.
        chunk_size (int): Number of samples per chunk.

    Returns:
        list: List of smaller chunks of training data.
    """
    return [training_data[i:i + chunk_size] for i in range(0, len(training_data), chunk_size)]


def generate_scene_embeddings(scene_data, model_name="paraphrase-MiniLM-L6-v2", batch_size=16, epochs=5, max_pairs_per_scene=50, chunk_size=50_000):
    """
    Incrementally fine-tune a SentenceTransformer model in chunks and generate embeddings for scene data.

    Args:
        scene_data (pd.DataFrame): DataFrame with scene details.
        model_name (str): Pre-trained SentenceTransformer model to fine-tune.
        batch_size (int): Number of samples to process in one batch.
        epochs (int): Number of training epochs.
        max_pairs_per_scene (int): Maximum number of pairs to generate per scene for fine-tuning.
        chunk_size (int): Number of training pairs per chunk.

    Returns:
        np.ndarray: Embeddings array.
    """
    try:
        logger.info("Preprocessing scene data for consistency...")
        columns_to_clean = ["processed_text", "tone"]
        scene_data = preprocess_scene_data_neutral(scene_data, columns_to_clean)

        if "processed_text" not in scene_data.columns:
            raise ValueError("Processed text column is missing in the scene_data DataFrame.")

        scene_texts = scene_data["processed_text"]
        if scene_texts.empty:
            logger.error("Scene texts are empty after preprocessing. Exiting embedding generation.")
            return

        logger.info(f"Starting incremental fine-tuning for {len(scene_texts)} scenes...")

        # Generate limited training data pairs
        logger.info("Generating limited training data pairs for fine-tuning...")
        training_data = []
        for i in range(len(scene_texts)):
            sampled_indices = random.sample(range(len(scene_texts)), min(max_pairs_per_scene, len(scene_texts) - 1))
            for j in sampled_indices:
                if i != j:
                    training_data.append(InputExample(texts=[scene_texts.iloc[i], scene_texts.iloc[j]], label=1.0))
            if len(training_data) % 100_000 == 0:
                logger.info(f"Generated {len(training_data)} training pairs so far...")

        if len(training_data) == 0:
            logger.error("No valid training pairs generated for fine-tuning. Check scene data.")
            return

        # Split training data into chunks
        logger.info(f"Splitting training data into chunks of size {chunk_size}...")
        chunks = [training_data[i:i + chunk_size] for i in range(0, len(training_data), chunk_size)]
        logger.info(f"Created {len(chunks)} chunks for incremental training.")

        # Initialize the model
        model = SentenceTransformer(model_name)
        train_loss = losses.CosineSimilarityLoss(model)

        # Incremental fine-tuning on chunks
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Fine-tuning on chunk {chunk_idx + 1}/{len(chunks)} with {len(chunk)} pairs...")
            train_dataloader = DataLoader(chunk, shuffle=True, batch_size=batch_size)
            model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=100)
            logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} fine-tuning completed.")

        logger.info("Fine-tuning completed for all chunks.")

        # Generate embeddings
        embeddings = []
        logger.info("Generating embeddings in batches...")
        for start in range(0, len(scene_texts), batch_size):
            batch_texts = scene_texts.iloc[start:start + batch_size].tolist()
            if not all(isinstance(text, str) for text in batch_texts):
                raise ValueError("Batch contains non-string values.")
            logger.info(f"Processing batch {start // batch_size + 1}/{(len(scene_texts) + batch_size - 1) // batch_size} "
                        f"with {len(batch_texts)} items...")
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=False)
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        # Validate dimensions
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            logger.error(f"Invalid embeddings: Expected a 2D numpy array, got {type(embeddings)} with ndim={getattr(embeddings, 'ndim', None)}")
            raise ValueError("Generated embeddings are not 2D. Ensure consistent dimensions.")

        # Save embeddings and metadata
        np.save(EMBEDDINGS_FILE, embeddings)
        logger.info(f"Embeddings saved to {EMBEDDINGS_FILE}")

        metadata = scene_data["imdb_id"].tolist()
        np.save(METADATA_FILE, metadata)
        logger.info(f"Metadata saved to {METADATA_FILE}")

        logger.info("Embedding generation completed successfully.")
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
        user_embedding = model.encode([user_scene], convert_to_tensor=True)

        # Validate embedding dimensions
        if user_embedding.ndim == 1:
            user_embedding = user_embedding.reshape(1, -1)
        elif user_embedding.ndim > 2:
            raise ValueError(f"User embedding has invalid dimensions: {user_embedding.shape}")

        # Compute cosine similarity
        logger.info("Computing cosine similarity...")
        similarities = cosine_similarity(user_embedding, embeddings)

        # Validate dimensions of similarities
        if similarities.ndim != 2 or similarities.shape[0] != 1:
            raise ValueError(f"Unexpected similarity dimensions: {similarities.shape}")

        similarities = similarities[0]  # Extract the first row

        # Sort by similarity and fetch top N
        top_indices = similarities.argsort()[-top_n:][::-1]
        seen_titles = set()  # Track seen IMDb IDs
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
    """
    Generate recommendations based on narrative content filtering.

    If `user_scene` is empty, fallback to recommending popular or trending movies.

    Args:
        engine: SQLAlchemy engine for database connection.
        user_scene (str): User-provided scene description.
        top_n (int): Number of recommendations to return.

    Returns:
        list of dict: Narrative-based recommendations with metadata.
    """
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
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        user_embedding = model.encode([user_scene], convert_to_tensor=False).reshape(1, -1)
        cosine_scores = cosine_similarity(user_embedding, embeddings)[0]

        # Sort by similarity and fetch top N
        top_indices = cosine_scores.argsort()[-top_n:][::-1]
        seen_titles = set()
        recommendations = [
            {"imdb_id": metadata[i], "score": cosine_scores[i]}
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
def generate_hybrid_recommendations(engine, user_scene, user_id, content_weight=0.4, collab_weight=0.6, top_n=5):
    """
    Generate hybrid recommendations combining content-based and interactive filtering.

    Args:
        engine: SQLAlchemy engine for database connection.
        user_scene (str): User-provided scene description.
        user_id (int): User ID for interactive filtering.
        content_weight (float): Weight for content-based filtering.
        collab_weight (float): Weight for interactive filtering.
        top_n (int): Number of recommendations to return.

    Returns:
        list of dict: Hybrid recommendations with metadata and combined scores.
    """
    try:
        # Load precomputed embeddings and metadata
        embeddings, metadata = load_embeddings()

        # Check for interactive filtering model
        if not os.path.exists(INTER_MODEL_FILE):
            logger.warning("Interactive filtering model not found. Training the model...")
            training_result = train_interactive_model(engine)
            if not training_result or training_result.get("message") != "Interactive model trained successfully!":
                return [{"message": "Failed to train interactive filtering model."}]

        with open(INTER_MODEL_FILE, "rb") as f:
            interactive_model = pickle.load(f)

        # Generate content-based scores
        model = SentenceTransformer("fine_tuned_model")  # Use the fine-tuned model
        user_embedding = model.encode([user_scene], convert_to_tensor=False).reshape(1, -1)
        cosine_scores = cosine_similarity(user_embedding, embeddings)[0]
        content_scores = {metadata[i]: cosine_scores[i] for i in range(len(metadata))}

        # Generate interactive filtering scores
        movie_query = text("SELECT DISTINCT imdb_id FROM movies")
        with engine.connect() as connection:
            movie_ids = [row[0] for row in connection.execute(movie_query)]
        interactive_scores = {
            movie_id: interactive_model.predict(user_id, movie_id).est for movie_id in movie_ids
        }

        # Combine content-based and interactive scores
        hybrid_scores = {
            imdb_id: content_weight * content_scores.get(imdb_id, 0) + collab_weight * interactive_scores.get(imdb_id, 0)
            for imdb_id in set(content_scores.keys()).union(interactive_scores.keys())
        }

        # Filter and rank recommendations
        seen_titles = set()
        filtered_scores = {
            imdb_id: score for imdb_id, score in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            if imdb_id not in seen_titles and not seen_titles.add(imdb_id)
        }

        # Fetch metadata for the top recommendations
        top_ids = list(filtered_scores.keys())[:top_n]
        metadata_query = text("""
            SELECT imdb_id, title, genre, overview, director
            FROM movies
            WHERE imdb_id = ANY(:top_ids)
        """)
        with engine.connect() as connection:
            metadata_results = connection.execute(metadata_query, {"top_ids": top_ids}).fetchall()

        if not metadata_results:
            logger.warning("No metadata found for the top recommended movies.")
            return [{"message": "No metadata available for recommendations."}]

        recommendations = []
        for row in metadata_results:
            try:
                recommendations.append({
                    "imdb_id": row["imdb_id"],
                    "title": row["title"],
                    "genre": row["genre"],
                    "overview": row["overview"],
                    "director": row["director"],
                    "score": filtered_scores.get(row["imdb_id"], 0),
                })
            except KeyError as e:
                logger.error(f"KeyError accessing metadata fields: {e}")
                continue

        return recommendations
    except Exception as e:
        logger.error(f"Error generating hybrid recommendations: {e}")
        return [{"message": "An error occurred while generating hybrid recommendations."}]
    
if __name__ == "__main__":
    def main():
        """
        Main function to generate the fine-tuned SentenceTransformer model
        and embeddings for narrative recommendations.
        """
        try:
            # Initialize database connection
            engine = get_sqlalchemy_engine()
            if not engine:
                logger.error("Failed to establish database connection. Exiting.")
                return

            # Fetch and preprocess scene data
            logger.info("Fetching and preprocessing scene data...")
            scene_data = fetch_scene_data(engine)
            if scene_data.empty:
                logger.error("No scene data available. Exiting.")
                return

            # Fine-tune the model and generate embeddings
            logger.info("Fine-tuning the SentenceTransformer model and generating embeddings...")
            generate_scene_embeddings(scene_data)

            logger.info("Model fine-tuning and embedding generation completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred during model fine-tuning: {e}")

    # Call the main function
    main()