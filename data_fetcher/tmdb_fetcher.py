import os
import sys
import logging
import requests
from time import sleep
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor

# Import configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import DATABASE, TMDB_API_KEY

# TMDb API
BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", TMDB_API_KEY)

# Configure logging
logging.basicConfig(
    filename="tmdb_fetcher.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set to INFO for general operation, DEBUG for troubleshooting
)

# Database Connection
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

# Fetch detailed movie metadata from TMDb API with retries
def get_detailed_metadata(movie_id, retries=3, delay=2):
    """
    Fetch detailed metadata for a movie from TMDb API with retry logic.

    Args:
        movie_id (int): TMDb movie ID.
        retries (int): Number of retry attempts.
        delay (int): Delay between retries in seconds.

    Returns:
        dict: Detailed metadata for the movie, or None if fetching fails.
    """
    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords"
            response = requests.get(url)

            if response.status_code == 429:  # Too many requests
                logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                sleep(delay)
                continue

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching detailed metadata for movie ID {movie_id} (Attempt {attempt + 1}): {e}")
        sleep(delay)
    return None

# Fetch movie metadata by title
def get_best_match(results, movie_title, release_year=None):
    """
    Choose the best match from TMDb results based on title and year.

    Args:
        results (list): List of TMDb search results.
        movie_title (str): Movie title to match.
        release_year (str): Optional release year for additional matching.

    Returns:
        dict: Best matching movie metadata.
    """
    for result in results:
        if result["title"].lower() == movie_title.lower():
            if release_year and str(result.get("release_date", "")).startswith(str(release_year)):
                return result
    return results[0] if results else None

def get_movie_metadata(movie_title, release_year=None):
    """
    Fetch movie metadata from TMDb API by title.

    Args:
        movie_title (str): Movie title.
        release_year (str): Optional release year.

    Returns:
        dict: Movie metadata or None if not found.
    """
    try:
        logging.info(f"Fetching metadata for movie title: {movie_title}")
        url = f"{BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url)
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            best_match = get_best_match(results, movie_title, release_year)
            return get_detailed_metadata(best_match["id"])
        else:
            logging.warning(f"No metadata found for movie title: {movie_title}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching metadata for movie title {movie_title}: {e}")
        return None

# Insert movie metadata into the movies table
def insert_movie_metadata(movie_data, engine):
    """
    Insert or update movie metadata in the database.

    Args:
        movie_data (dict): Movie metadata.
        engine: SQLAlchemy engine.
    """
    try:
        query = text("""
            INSERT INTO movies (movie_id, title, year, genre, overview, director, runtime, keywords)
            VALUES (:movie_id, :title, :year, :genre, :overview, :director, :runtime, :keywords)
            ON CONFLICT (movie_id) DO UPDATE SET
                title = EXCLUDED.title,
                year = EXCLUDED.year,
                genre = EXCLUDED.genre,
                overview = EXCLUDED.overview,
                director = EXCLUDED.director,
                runtime = EXCLUDED.runtime,
                keywords = EXCLUDED.keywords
        """)

        movie_id = movie_data.get("id")
        title = movie_data.get("title", "Unknown Title")
        year = movie_data.get("release_date", "N/A").split("-")[0] if "release_date" in movie_data else "N/A"
        genres = movie_data.get("genres", [])
        genre = ", ".join([g.get("name", "Unknown Genre") for g in genres])
        overview = movie_data.get("overview", "No overview available")
        director = next(
            (c.get("name") for c in movie_data.get("credits", {}).get("crew", []) if c.get("job") == "Director"),
            "Unknown Director"
        )
        runtime = movie_data.get("runtime", "Unknown")
        keywords = ", ".join([k.get("name", "Unknown Keyword") for k in movie_data.get("keywords", {}).get("keywords", [])])

        with engine.connect() as connection:
            connection.execute(query, {
                "movie_id": movie_id,
                "title": title,
                "year": year,
                "genre": genre,
                "overview": overview,
                "director": director,
                "runtime": runtime,
                "keywords": keywords
            })
        logging.info(f"Inserted/Updated movie metadata for {title} ({year})")
    except Exception as e:
        logging.error(f"Error inserting movie metadata: {e}")

# Link movie ID to scripts in the database
def link_movie_metadata_to_scripts(movie_data, engine):
    """
    Link movie metadata to scripts in the database.

    Args:
        movie_data (dict): Movie metadata.
        engine: SQLAlchemy engine.
    """
    try:
        movie_id = movie_data.get("id")
        title = movie_data.get("title", "").strip()
        if not movie_id or not title:
            logging.warning(f"Skipping linking for invalid movie data: {movie_data}")
            return

        query = text("""
            UPDATE {table}
            SET movie_id = :movie_id
            WHERE TRIM(LOWER(title)) = TRIM(LOWER(:title))
        """)

        with engine.connect() as connection:
            for table in ["three_act_scripts", "nine_act_scripts"]:
                connection.execute(query.format(table=text(table)), {"movie_id": movie_id, "title": title})

        logging.info(f"Linked movie_id ({movie_id}) to scripts for title: {title}")
    except Exception as e:
        logging.error(f"Error linking movie metadata to scripts for {title}: {e}")

# Fetch and link movie metadata and scripts in parallel
def process_movie(title, engine):
    """
    Process a single movie: fetch metadata, insert it, and link it to scripts.

    Args:
        title (str): Movie title.
        engine: SQLAlchemy engine.
    """
    try:
        movie_data = get_movie_metadata(title)
        if movie_data:
            insert_movie_metadata(movie_data, engine)
            link_movie_metadata_to_scripts(movie_data, engine)
        else:
            logging.warning(f"No metadata found for title: {title}")
    except Exception as e:
        logging.error(f"Error processing movie '{title}': {e}")

def fetch_and_link_metadata():
    """
    Fetch movie titles from scripts tables and link metadata in parallel.
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logging.error("No database connection. Skipping metadata fetching.")
        return

    try:
        with engine.connect() as connection:
            query = text("""
                SELECT DISTINCT title FROM three_act_scripts
                WHERE title IS NOT NULL
                UNION
                SELECT DISTINCT title FROM nine_act_scripts
                WHERE title IS NOT NULL
            """)
            movie_titles = [row[0] for row in connection.execute(query).fetchall()]

        if not movie_titles:
            logging.warning("No movie titles found in the scripts tables.")
            return

        with ThreadPoolExecutor(max_workers=5) as executor:
            for title in movie_titles:
                executor.submit(process_movie, title, engine)
    except Exception as e:
        logging.error(f"Error fetching metadata for scripts: {e}")