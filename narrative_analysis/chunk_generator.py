import re
import logging
import unicodedata

# Logging configuration
logging.basicConfig(
    filename="chunk_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def preprocess_line(line):
    """
    Preprocess and clean script lines to remove noise and ensure readability.
    """
    try:
        # Validate input
        if not isinstance(line, str):
            logging.warning(f"Invalid input to preprocess_line: {type(line)} detected. Skipping line.")
            return ""

        # Normalize Unicode to ASCII
        line = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode("utf-8", "ignore")

        # Remove standalone line numbers (e.g., "144", "135.")
        line = re.sub(r"^\s*\d+\.?\s*$", "", line)  # Matches standalone numbers with or without a trailing period
        
        # Remove introductory metadata
        metadata_patterns = [
            r"FOR YOUR CONSIDERATION",              # Common promotional header
            r"OUTSTANDING ORIGINAL SCREENPLAY",     # Award-specific text
            r"(SCREENPLAY BY|STORY BY|WRITTEN BY).*",  # Writing credits
            r"^\s*[A-Z ]+\s*$",                     # Lines with only uppercase words (e.g., movie titles)
            r"NEON",                                # Production company
            r"^\s*BY\s[A-Z\s]+$",                   # Generic "BY AUTHOR(S)" lines
            r"PRODUCED BY.*",                       # Production credits
            r"DIRECTED BY.*",                       # Director credits
            r"COPYRIGHT .*",                        # Copyright information
            r"PRESENTED BY.*",                      # Presenter credits
            r"\f"                                   # Form feed characters
        ]
        for pattern in metadata_patterns:
            line = re.sub(pattern, "", line, flags=re.IGNORECASE).strip()

        # Remove HTML tags and entities
        line = re.sub(r"<.*?>", "", line)  # HTML tags
        line = re.sub(r"&[a-z]+;", "", line)  # HTML entities

        # Remove excessive whitespace
        line = re.sub(r"\s+", " ", line)

        # Normalize capitalization and spacing
        line = line.capitalize() if line.isupper() else line
        line = re.sub(r"([A-Z]{2,})\s([A-Z]{2,})", r"\1 \2", line)  # Avoid concatenated uppercase words
        line = re.sub(r"\s([.,!?])", r"\1", line)  # Remove space before punctuation

        # Handle parentheticals, including nested ones
        if "(" in line and ")" in line:
            line = re.sub(r"\(([^()]*)\)", r"[\1]", line)

        # Normalize repeated special characters
        line = re.sub(r"([!?.])\1+", r"\1", line)  # Replace multiple "!!!" with "!"

        # Remove non-alphanumeric characters, except basic punctuation
        line = re.sub(r"[^a-zA-Z0-9.,!?\'\s\[\]\-]", "", line)

        return line.strip()
    except Exception as e:
        logging.error(f"Error during preprocessing of line: {e}")
        return ""

def postprocess_line(chunk_text):
    """
    Beautify the chunk text to make it more readable and structured for database storage.
    
    Args:
        chunk_text (str): The raw chunk text.
    
    Returns:
        str: The beautified chunk text.
    """
    try:
        # Remove excessive whitespace
        chunk_text = re.sub(r"\s+", " ", chunk_text).strip()

        # Add line breaks before scene headers (e.g., INT., EXT.)
        chunk_text = re.sub(r"(?<=\.\s)(INT\.|EXT\.)", r"\n\n\1", chunk_text)

        # Add line breaks before and after speaker dialogues (e.g., "Chung-sook:")
        chunk_text = re.sub(r"([A-Za-z\-\']+):", r"\n\n\1:", chunk_text)

        # Add line breaks before major transitions (e.g., "LIVING ROOM SLASH KITCHEN")
        chunk_text = re.sub(r"(?<=\.\s)([A-Z\s\-\/]+)\s+", r"\n\n\1\n", chunk_text)

        # Standardize [contd], [O.S.], [V.O.] with brackets
        chunk_text = re.sub(r"\[contd\]", "[CONT'D]", chunk_text, flags=re.IGNORECASE)
        chunk_text = re.sub(r"\[o\.s\.\]", "[O.S.]", chunk_text, flags=re.IGNORECASE)
        chunk_text = re.sub(r"\[v\.o\.\]", "[V.O.]", chunk_text, flags=re.IGNORECASE)

        # Ensure proper spacing after punctuation
        chunk_text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", chunk_text)

        # Ensure consistent capitalization for scene headers
        chunk_text = re.sub(r"(INT\.|EXT\.)", lambda m: m.group(1).upper(), chunk_text)

        # Trim and finalize formatting
        chunk_text = chunk_text.strip()

        return chunk_text
    except Exception as e:
        logging.error(f"Error beautifying chunk text: {e}")
        return chunk_text  # Return the original chunk_text

def normalize_and_chunk(script_text):
    """
    Normalize and split raw script into meaningful chunks for act detection,
    with enhanced fallback and error handling for edge cases.
    """
    try:
        # Validate input
        if not isinstance(script_text, str) or not script_text.strip():
            logging.error("Invalid input to normalize_and_chunk: script_text must be a non-empty string.")
            return []

        # Patterns for identifying irrelevant lines
        page_number_patterns = [
            r"^\s*page \d+\s*$",  # "Page 12"
            r"^\s*\d+\s*/\s*\d+\s*$",  # "12/130"
            r"^\s*\d+\s*$",  # "12"
            r"^\s*page \d+\s*of\s*\d+\s*$",  # "Page 12 of 130"
            r"^\s*page\s*\d+\s*\(\s*\d+\s*\)$",  # "Page 12 (12)"
            r"^\s*pg\s*\d+\s*$",  # "Pg 12"
        ]

        revision_date_patterns = [
            r"^\s*[a-zA-Z]+\s+\(\d{1,2}/\d{1,2}/\d{2,4}\)\s*$",  # "June (6/5/2019)"
            r"^\s*\(rev\.\s+\d{1,2}/\d{1,2}/\d{2,4}\)\s*$",  # "(Rev. 6/5/2019)"
            r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$",  # "6/5/2019"
            r"^\s*\(updated\s+\d{1,2}/\d{1,2}/\d{2,4}\)\s*$",  # "(Updated 6/5/2019)"
            r"^\s*\w+\s+draft\s+\d{1,2}/\d{1,2}/\d{2,4}\s*$",  # "Final Draft 6/5/2019"
        ]

        scene_header_patterns = [
            r"^(INT\.|EXT\.|INT/EXT\.|I/E|ESTABLISHING|FADE IN|FADE OUT|CUT TO|DISSOLVE TO)",  # Core scene headers
            r"^(BLACK SCREEN|TITLE CARD|OPENING SHOT|POV|CLOSE UP|OVER THE SHOULDER|ANGLE ON|CAPTION)",  # Scene-setting terms
            r"^(SCENE \d+:|SHOT \d+:)",  # Scene/Shot numbering
            r"^\s*[A-Z]+.*(- DAY|- NIGHT|DAY|NIGHT)\s*$",  # Location with time of day
            r"^\s*(MONTAGE|SERIES OF SHOTS).*$",  # Montages or shot sequences
            r"^\s*(FLASHBACK|FLASHFORWARD).*$",  # Flashbacks/flashforwards
            r"^\s*(SUPERIMPOSE|TEXT ON SCREEN|TRANSITION TO).*$",  # Additional transitions
            r"^\s*CLOSE ON.*$",  # Close-up shots
        ]

        # Initialize chunking
        lines = script_text.split("\n")
        chunks = []
        current_chunk = []
        discarded_chunks = []
        min_chunk_length = 128  # Minimum chunk length to prevent overly small chunks
        max_chunk_length = int(0.85 * 12288)  # 85% of GPT token limit for safety margin

        for line in lines:
            try:
                # Preprocess the line
                line = preprocess_line(line)
            except Exception as e:
                logging.error(f"Error preprocessing line: {e}")
                continue

            # Skip empty or invalid lines
            if not line.strip():
                continue

            # Ignore page numbers and revision dates
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in page_number_patterns + revision_date_patterns):
                logging.debug(f"Ignoring line: {line}")
                continue

            # Handle scene headers
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in scene_header_patterns):
                if current_chunk:
                    if len(" ".join(current_chunk)) >= min_chunk_length:
                        chunks.append(" ".join(current_chunk).strip())
                    else:
                        discarded_chunks.append(" ".join(current_chunk).strip())
                    current_chunk = []
                current_chunk.append(line)
                continue
            # Add line to current chunk
            current_chunk.append(line)

            # Split chunk if it exceeds max length
            if len(" ".join(current_chunk)) > max_chunk_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Add the remaining chunk
        if current_chunk:
            if len(" ".join(current_chunk)) >= min_chunk_length:
                chunks.append(" ".join(current_chunk))
            else:
                discarded_chunks.append(" ".join(current_chunk))

        # Merge discarded chunks intelligently
        for discarded in discarded_chunks:
            if not chunks:  # If no chunks exist, create the first chunk
                chunks.append(discarded)
            elif len(chunks[-1]) < max_chunk_length:
                # Prefer merging with the previous chunk if space allows
                chunks[-1] += f" {discarded}"
            else:
                # Merge with the next chunk if space allows, or add as a standalone chunk
                if len(discarded) + (len(chunks[0]) if chunks else 0) <= max_chunk_length:
                    chunks.append(discarded)
                else:
                    chunks[-1] += f" {discarded}"

        return chunks
    except Exception as e:
        logging.error(f"Error in normalize_and_chunk: {e}")
        return []
    
def generate_chunks(raw_script):
    """
    Generate chunks from raw script.

    Args:
        raw_script (str): The raw screenplay text.

    Returns:
        list of dict: List of chunks with chunk_number and chunk_text.
    """
    chunk_texts = normalize_and_chunk(raw_script)
    return [{"chunk_number": i + 1, "chunk_text": text} for i, text in enumerate(chunk_texts)]