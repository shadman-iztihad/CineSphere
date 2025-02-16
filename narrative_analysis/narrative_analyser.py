import os
import sys
import json
import time
import logging
import pandas as pd
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from analyser_main import enrich_chunks
from chunk_generator import postprocess_line, generate_chunks
from concurrent.futures import ThreadPoolExecutor, as_completed

# Database and OpenAI configurations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import DATABASE

# Configure logging
logging.basicConfig(
    filename="narrative_analyser.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

def get_processed_imdb_ids(engine):
    """
    Fetch all IMDb IDs already processed in the nine_act_annotations table.

    Args:
        engine: SQLAlchemy database engine.

    Returns:
        set: A set of IMDb IDs that are already processed.
    """
    try:
        query = "SELECT DISTINCT imdb_id FROM nine_act_annotations;"
        with engine.connect() as connection:
            result = connection.execute(text(query))
            # Access rows using tuple indexing
            processed_ids = {row[0] for row in result}  # Use index 0 to access imdb_id
        logging.info(f"Fetched {len(processed_ids)} processed IMDb IDs from the database.")
        return processed_ids
    except Exception as e:
        logging.error(f"Error fetching processed IMDb IDs: {e}")
        return set()

def fetch_scripts(engine, limit=10, offset=0, exclude_ids=None):
    """
    Fetch a batch of raw scripts from the screenplays table, excluding already processed IMDb IDs.

    Args:
        engine: SQLAlchemy database engine.
        limit (int): Number of scripts to fetch in a single batch.
        offset (int): Starting point for fetching scripts.
        exclude_ids (set): IMDb IDs to exclude from the fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched scripts.
    """
    exclude_clause = ""
    if exclude_ids:
        exclude_ids_list = ",".join(f"'{imdb_id}'" for imdb_id in exclude_ids)
        exclude_clause = f"AND imdb_id NOT IN ({exclude_ids_list})"

    query = f"""
        SELECT imdb_id, raw_script
        FROM screenplays
        WHERE raw_script IS NOT NULL {exclude_clause}
        ORDER BY imdb_id ASC
        LIMIT {limit} OFFSET {offset};
    """
    try:
        df = pd.read_sql_query(query, con=engine)
        logging.info(f"Fetched {len(df)} screenplays starting at offset {offset}, excluding processed IDs.")
        return df
    except Exception as e:
        logging.error(f"Error fetching screenplays: {e}")
        return pd.DataFrame()
    
DAILY_REQUEST_LIMIT = 40  # Maximum number of requests allowed per day
PROGRESS_FILE = "processing_progress.json"  # File to save progress

def load_progress():
    """
    Load the progress from the progress file.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as file:
            return json.load(file)
    return {"offset": 0, "completed_scripts": 0, "requests_made": 0}

def save_progress(offset, completed_scripts, requests_made):
    """
    Save the progress to the progress file.
    """
    progress = {
        "offset": offset,
        "completed_scripts": completed_scripts,
        "requests_made": requests_made
    }
    with open(PROGRESS_FILE, "w") as file:
        json.dump(progress, file)
    logging.info(f"Progress saved: {progress}")

def save_annotations(engine, target_table, imdb_id, annotations):
    """
    Save screenplay chunk analysis results to the database and log validation errors.

    Args:
        engine: SQLAlchemy database engine.
        target_table (str): The name of the target table (e.g., 'nine_act_annotations').
        imdb_id (str): The IMDb ID of the movie being processed.
        annotations (list of dict): List of annotations, where each annotation contains:
            - "chunk_number" (int): The chunk number.
            - "chunk_text" (str): The text of the chunk.
            - "action_summary" (str): Key actions/events summary.
            - "tone" (str): Emotional tone of the chunk.
            - "key_elements" (str): Critical props/symbols/settings.
            - "thematic_analysis" (str): Analysis of themes and character development.
    """
    required_keys = {
        "chunk_number", "chunk_text", "action_summary", "tone", "key_elements", "thematic_analysis"
    }

    try:
        if not isinstance(annotations, list) or not all(isinstance(annotation, dict) for annotation in annotations):
            raise ValueError("Annotations must be a list of dictionaries.")

        # Validate annotations and log errors for missing keys
        error_log_path = "validation_errors.log"
        with open(error_log_path, "a") as error_log:
            for idx, annotation in enumerate(annotations):
                missing_keys = required_keys - annotation.keys()
                if missing_keys:
                    # Log the error message for missing keys
                    error_message = (
                        f"IMDb ID {imdb_id}, Annotation {idx + 1}: Missing keys: {missing_keys}"
                    )
                    logging.error(error_message)
                    error_log.write(f"{error_message}\n")

                    # Assign default values for the missing keys
                    for key in missing_keys:
                        annotation[key] = "N/A"

        with engine.begin() as connection:
            # Prepare bulk data for insertion
            insert_data = [
                {
                    "imdb_id": imdb_id,
                    "chunk_number": annotation["chunk_number"],
                    "chunk_text": postprocess_line(annotation["chunk_text"]),
                    "action_summary": annotation.get("action_summary", "N/A"),
                    "tone": json.dumps(annotation.get("tone", "N/A")),  # Serialize tone to JSON
                    "key_elements": json.dumps(annotation.get("key_elements", "N/A")),  # Serialize key_elements to JSON
                    "thematic_analysis": annotation.get("thematic_analysis", "N/A"),
                }
                for annotation in annotations
            ]

            # Log the processed data
            logging.debug(f"Data to be inserted for IMDb ID {imdb_id}: {insert_data}")

            # SQL bulk insert query
            query = text(f'''
                INSERT INTO {target_table} 
                (imdb_id, chunk_number, chunk_text, action_summary, tone, key_elements, thematic_analysis)
                VALUES (:imdb_id, :chunk_number, :chunk_text, :action_summary, :tone, :key_elements, :thematic_analysis)
                ON CONFLICT (imdb_id, chunk_number) DO UPDATE
                SET 
                    action_summary = EXCLUDED.action_summary,
                    tone = EXCLUDED.tone,
                    key_elements = EXCLUDED.key_elements,
                    thematic_analysis = EXCLUDED.thematic_analysis;
            ''')
            connection.execute(query, insert_data)
    except Exception as e:
        logging.error(f"Error saving annotations for IMDb ID {imdb_id}: {e}")

def process_single_script(engine, screenplay, current_idx, total_screenplays):
    imdb_id = screenplay["imdb_id"]
    raw_script = screenplay["raw_script"]

    try:
        logging.info(f"[IMDb ID {imdb_id}] Starting processing ({current_idx}/{total_screenplays}).")

        # Generate chunks
        chunks = generate_chunks(raw_script)
        if not chunks:
            logging.warning(f"[IMDb ID {imdb_id}] No chunks generated. Skipping.")
            return

        total_chunks = len(chunks)
        logging.info(f"[IMDb ID {imdb_id}] Generated {total_chunks} chunks.")

        # Enrich and save chunks incrementally
        for idx, chunk in enumerate(chunks, start=1):
            try:
                # Log enrichment progress
                logging.info(f"[IMDb ID {imdb_id}] Enriching chunk {idx}/{total_chunks}.")
                
                # Enrich the chunk and save it
                enriched_chunk = enrich_chunks([chunk])[0]
                save_annotations(engine, "nine_act_annotations", imdb_id, [enriched_chunk])
                
                # Log successful save
                logging.info(f"[IMDb ID {imdb_id}] Chunk {idx}/{total_chunks} enriched and saved.")
            except Exception as e:
                logging.error(f"[IMDb ID {imdb_id}] Error enriching or saving chunk {chunk['chunk_number']}: {e}")

        logging.info(f"[IMDb ID {imdb_id}] Completed processing ({current_idx}/{total_screenplays}). Total enriched chunks: {total_chunks}.")
    except Exception as e:
        logging.error(f"[IMDb ID {imdb_id}] Error processing screenplay: {e}")

def process_screenplays(batch_size=10, delay_between_batches=30, total_scripts=1362):
    """
    Main function to process screenplays in batches, with de-duplication and progress tracking.

    Args:
        batch_size (int): Number of movies to process in each batch.
        total_scripts (int): Total number of scripts to process (before de-duplication).
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logging.error("Failed to connect to the database.")
        return

    # Load progress
    progress = load_progress()
    offset = progress["offset"]
    completed_scripts = progress["completed_scripts"]
    requests_made = progress["requests_made"]

    # Fetch already processed IMDb IDs
    processed_ids = get_processed_imdb_ids(engine)

    # Adjust total_scripts to reflect unprocessed movies
    unprocessed_count = total_scripts - len(processed_ids)
    logging.info(f"Fetched {len(processed_ids)} processed IMDb IDs from the database.")
    logging.info(f"Total unprocessed movies remaining: {unprocessed_count}.")
    total_scripts = unprocessed_count

    total_batches = (total_scripts + batch_size - 1) // batch_size  # Calculate total batches

    while requests_made < DAILY_REQUEST_LIMIT:
        batch_number = (offset // batch_size) + 1  # Current batch number
        screenplays = fetch_scripts(engine, limit=batch_size, offset=offset, exclude_ids=processed_ids)
        if screenplays.empty:
            logging.info("No more screenplays to process. All batches completed.")
            break

        batch_size_actual = len(screenplays)
        logging.info(f"Processing batch {batch_number}/{total_batches} starting at offset {offset} with {batch_size_actual} screenplays...")

        max_workers = min(batch_size_actual, 4)  # Adjust workers dynamically

        # Process screenplays in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_script, engine, screenplay, idx + 1 + completed_scripts, total_scripts)
                for idx, screenplay in screenplays.iterrows()
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                    completed_scripts += 1
                    requests_made += 1  # Increment request count
                    logging.info(f"Progress: {completed_scripts}/{total_scripts} screenplays processed. Requests made: {requests_made}.")

                    # Stop if daily limit is reached
                    if requests_made >= DAILY_REQUEST_LIMIT:
                        logging.warning("Daily request limit reached. Stopping processing.")
                        save_progress(offset, completed_scripts, requests_made)
                        return
                except Exception as e:
                    logging.error(f"Error in parallel processing: {e}")

        # Increment offset and delay between batches
        offset += batch_size
        if not screenplays.empty:
            logging.info(f"Batch {batch_number}/{total_batches} completed. Waiting {delay_between_batches} seconds before starting the next batch...")
            time.sleep(delay_between_batches)

    save_progress(offset, completed_scripts, requests_made)
    logging.info(f"All {completed_scripts}/{total_scripts} screenplays processed successfully.")

if __name__ == "__main__":
    process_screenplays()