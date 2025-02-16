import re
import os
import sys
import json
import PyPDF2
import chardet
import logging
import requests
import psycopg2
from psycopg2 import sql
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from time import sleep

# Add backend config path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import DATABASE  # Import database config

# Configure logging
logging.basicConfig(
    filename="manual_scripting.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Set to INFO for general operation, DEBUG for troubleshooting
)

# Constants
DOWNLOAD_DIR = "Scripts"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
MAX_WORKERS = 5
RETRIES = 3
RETRY_DELAY = 2

# Database connection
def get_db_connection():
    """Establish and return a database connection."""
    try:
        return psycopg2.connect(
            dbname=DATABASE["DB_NAME"],
            user=DATABASE["USER"],
            password=DATABASE["PASSWORD"],
            host=DATABASE["HOST"],
            port=DATABASE["PORT"]
        )
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None

# Load movie scripts from a JSON file
def load_movie_scripts(json_file):
    try:
        with open(json_file, "r") as file:
            data = json.load(file)
            if "three_act_scripts" in data:
                return data.get("three_act_scripts", []), "three_act_scripts"
            elif "nine_act_scripts" in data:
                return data.get("nine_act_scripts", []), "nine_act_scripts"
            else:
                logging.error(f"Invalid structure in JSON file: {json_file}")
                return [], None
    except Exception as e:
        logging.error(f"Error loading JSON file {json_file}: {e}")
        return [], None

# Download a movie script
def download_script(movie):
    """Download a movie script from the given URL with retries."""
    for attempt in range(RETRIES):
        try:
            response = requests.get(movie["url"], stream=True)
            if response.status_code == 200:
                sanitized_title = re.sub(r"[^\w\s-]", "", movie["title"]).replace(" ", "_")
                file_extension = movie.get("file_type", "pdf")  # Defaulting to PDF
                file_path = os.path.join(DOWNLOAD_DIR, f"{sanitized_title}.{file_extension}")

                with open(file_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)

                logging.info(f"Downloaded: {movie['title']}")
                return file_path
            else:
                logging.warning(f"Failed to download {movie['title']}: {response.status_code}")
        except Exception as e:
            logging.error(f"Error downloading {movie['title']} (Attempt {attempt + 1}): {e}")
        sleep(RETRY_DELAY)
    return None

# Modular script extractor
class ScriptExtractor:
    @staticmethod
    def extract(file_path, file_type):
        """Dynamically extract text based on the file type."""
        if file_type == "pdf":
            return ScriptExtractor.extract_pdf(file_path)
        elif file_type == "html":
            return ScriptExtractor.extract_html(file_path)
        elif file_type == "txt":
            return ScriptExtractor.extract_txt(file_path)
        else:
            logging.error(f"Unsupported file type: {file_type}")
            return None

    @staticmethod
    def extract_pdf(file_path):
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join([page.extract_text() or "" for page in reader.pages])
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF {file_path}: {e}")
            return None

    @staticmethod
    def extract_html(file_path):
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read()
                detected_encoding = chardet.detect(raw_data)['encoding']

            with open(file_path, "r", encoding=detected_encoding, errors="ignore") as file:
                soup = BeautifulSoup(file, "html.parser")
                raw_script = soup.get_text(separator="\n")
            return raw_script
        except Exception as e:
            logging.error(f"Error extracting text from HTML {file_path}: {e}")
            return None

    @staticmethod
    def extract_txt(file_path):
        try:
            with open(file_path, "rb") as file:
                raw_data = file.read()
                detected_encoding = chardet.detect(raw_data)['encoding']
            with open(file_path, "r", encoding=detected_encoding) as file:
                text = file.read()
            return text
        except Exception as e:
            logging.error(f"Error extracting text from TXT {file_path}: {e}")
            return None

# Validate script content
def validate_script(raw_script):
    """Check if a script contains sufficient content."""
    if len(raw_script.split()) < 50:
        logging.warning("Script is too short or empty.")
        return False
    return True

# Insert raw scripts into the database
def insert_raw_script_into_db(table_name, title, raw_script):
    conn = get_db_connection()
    if not conn:
        logging.error("No database connection. Skipping insertion.")
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            sql.SQL("""
                INSERT INTO {table} (title, raw_script)
                VALUES (%s, %s)
                ON CONFLICT (title) DO NOTHING
            """).format(table=sql.Identifier(table_name)),
            (title, raw_script)
        )
        conn.commit()
        logging.info(f"Inserted raw script for: {title} into {table_name}")
    except Exception as e:
        logging.error(f"Error inserting raw script for {title} into {table_name}: {e}")
    finally:
        conn.close()

# Process a single movie
def process_movie(movie, table_name):
    file_path = download_script(movie)
    if not file_path:
        logging.warning(f"Skipping {movie['title']} due to download failure.")
        return

    file_type = movie.get("file_type", "pdf")  # Default to PDF
    raw_script = ScriptExtractor.extract(file_path, file_type)
    if raw_script and validate_script(raw_script):
        insert_raw_script_into_db(table_name, movie["title"], raw_script)
    else:
        logging.warning(f"Failed to process script for {movie['title']}")

# Process all scripts in a JSON file concurrently
def process_scripts(json_file):
    movie_scripts, table_name = load_movie_scripts(json_file)
    if not movie_scripts or not table_name:
        logging.error(f"Failed to load scripts or table name from {json_file}.")
        return

    print(f"Processing scripts for table: {table_name}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(lambda movie: process_movie(movie, table_name), movie_scripts)

if __name__ == "__main__":
    # print("Processing 3-act movie scripts...")
    # process_scripts("3_act_scripts.json")
    print("Processing 9-act movie scripts...")
    process_scripts("9_act_scripts.json")