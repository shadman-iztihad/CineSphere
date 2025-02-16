import os
import sys
import json
import time
import logging
from openai import OpenAI
from tiktoken import encoding_for_model

# Configure logging
logging.basicConfig(
    filename="analyser_main.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database and OpenAI configurations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(
    filename="csv_enrich.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# OpenAI API URL
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
DELAY = 4

def estimate_tokens(text, model="gpt-4o-mini"):
    """
    Estimate the number of tokens in the given text.

    Args:
        text (str): Input text.
        model (str): OpenAI model to use for token estimation.

    Returns:
        int: Number of tokens.
    """
    tokenizer = encoding_for_model(model)
    return len(tokenizer.encode(text))

def call_openai_api(chunk_number, chunk_text):
    """
    Make the API call to OpenAI to analyze screenplay chunks.
    """
    model = "gpt-4o-mini"
    token_limit = 16384  # Adjust based on max_tokens allowed
    prompt = f"""
    You are a screenplay analysis expert specializing the depth of storytelling in movie scripts. Analyze the following screenplay chunk and provide a detailed response in JSON format with the following fields:
    
    1. **action_summary**: Provide a clear, concise summary of the main actions or interactions in the chunk. This field is mandatory.
    2. **tone**: Use 3–5 adjectives to summarize the emotional tone.
    3. **key_elements**: Highlight the most critical props, symbols, or settings.
    4. **thematic_analysis**: Briefly explain the chunk’s thematic role and how it supports character development or the story arc.

    Chunk Details:
    Text: {chunk_text}
    """
    prompt_tokens = estimate_tokens(prompt, model=model)
    max_tokens = token_limit - prompt_tokens - 2000
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert in screenplay structure."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.5
    }

    delay = 2  # Initial delay for exponential backoff

    for attempt in range(3):
        try:
            response = client.chat.completions.create(**payload)
            if not response or not hasattr(response, "choices") or not response.choices:
                raise ValueError(f"Empty or malformed GPT response for chunk {chunk_number}.")

            message = response.choices[0].message
            if not message or not hasattr(message, "content"):
                raise ValueError(f"No content found in GPT response for chunk {chunk_number}.")

            raw_content = message.content
            logging.debug(f"Raw API Response for chunk {chunk_number}: {raw_content}")
            parsed_content = parse_json_content(raw_content)

            # Check for missing keys and retry if necessary
            required_keys = ["action_summary", "tone", "key_elements", "thematic_analysis"]
            for key in required_keys:
                if key not in parsed_content:
                    logging.warning(f"Missing key '{key}' in parsed content for chunk {chunk_number}. Retrying...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)  # Exponential backoff with cap
                    continue  # Retry the API call

            return parsed_content

        except Exception as e:
            if "rate limit" in str(e).lower():  # Check for rate limit errors
                logging.warning(f"Rate limit reached, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                delay = min(delay * 2, 60)  # Exponential backoff with cap
            else:
                logging.error(f"Unhandled error during GPT request for chunk {chunk_number}: {e}")
                break  # Exit retry loop for other errors

    if attempt + 1 == 3:
        # Return fallback values if retries are exhausted
        return {
            "action_summary": "Failed to generate action summary.",
            "tone": "N/A",
            "key_elements": "N/A",
            "thematic_analysis": "N/A",
        }

def parse_json_content(raw_content):
    try:
        # Clean and parse the JSON content
        cleaned_content = raw_content.strip("```json").strip("```").strip()
        parsed_content = json.loads(cleaned_content)

        # Ensure all required keys are present with fallback values
        required_keys = ["action_summary", "tone", "key_elements", "thematic_analysis"]
        for key in required_keys:
            if key not in parsed_content:
                logging.warning(f"Missing key '{key}' in parsed content. Adding default value.")
                parsed_content[key] = "N/A"

        return parsed_content
    except json.JSONDecodeError as e:
        # Log error and raw content for debugging
        logging.error(f"JSON parsing error: {e}. Content: {raw_content[:100]}")
        logging.debug(f"Full raw content: {raw_content}")

        # Return fallback values for all keys
        return {
            "action_summary": "Unable to parse.",
            "tone": "Unknown",
            "key_elements": "Unavailable",
            "thematic_analysis": "Parsing failed.",
        }

def enrich_chunks(chunks):
    """
    Enrich screenplay chunks using OpenAI API.

    Args:
        chunks (list of dict): List of chunks with chunk_number and chunk_text.

    Returns:
        list of dict: Enriched data for each chunk.
    """
    enriched_data = []
    required_keys = {"action_summary", "tone", "key_elements", "thematic_analysis"}

    for idx, chunk in enumerate(chunks, start=1):
        chunk_number = chunk["chunk_number"]
        chunk_text = chunk["chunk_text"]

        # Call OpenAI API to enrich the chunk
        analysis = call_openai_api(chunk_number, chunk_text)

        # Ensure all required keys are present in the analysis
        for key in required_keys:
            analysis[key] = analysis.get(key, "N/A")

        # Append enrichment fields to the chunk
        enriched_data.append({
            "chunk_number": chunk_number,
            "chunk_text": chunk_text,
            **analysis,
        })
        
    return enriched_data