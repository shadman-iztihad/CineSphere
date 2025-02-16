from dotenv import load_dotenv
import os

load_dotenv()

DATABASE = {
    "DB_NAME": os.getenv("DB_NAME", ""),
    "USER": os.getenv("DB_USER", ""),
    "PASSWORD": os.getenv("DB_PASSWORD", ""),
    "HOST": os.getenv("DB_HOST", ""),
    "PORT": os.getenv("DB_PORT", ""),
}

# TMDB API Key
TMDB_API_KEY=os.getenv("TMDB_API_KEY", "")

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")

GROQ_API_KEY=os.getenv("GROQ_API_KEY", "")

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY", "")