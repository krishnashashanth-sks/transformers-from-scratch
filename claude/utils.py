from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re

DetectorFactory.seed = 0

def clean_text_initial(text):
    if not isinstance(text, str):
        return ""

    # 1. Remove HTML/XML tags
    text = re.sub(r'<.*?>', '', text)

    # 2. Replace special characters or unwanted non-alphanumeric characters with a space.
    # This regex keeps alphanumeric characters, basic punctuation (.,!?-), single quotes, double quotes and spaces.
    # Adjust this regex based on what punctuation is considered meaningful for the LLM.
    text = re.sub(r'[^a-zA-Z0-9.,!?”"\'\s]', ' ', text)

    # 3. Normalize whitespace (multiple spaces, tabs, newlines to single space, then strip)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def is_english(text, threshold=0.9):
    try:
        # Langdetect can raise an exception for very short strings or non-detectable languages
        if len(text.split()) < 3: # Heuristic: require at least 3 words for detection
            return False
        detections = detect(text)
        # For simplicity, we'll just check if 'en' is the detected language
        # For more robust solutions, one might check `detect_langs` and a confidence score
        return detections == 'en'
    except LangDetectException:
        return False
    except Exception as e:
        # Catch other potential errors, e.g., if text is not a string
        # print(f"Error during language detection: {e}") # Uncomment for debugging
        return False
    
def filter_low_quality_content(text, min_alpha_ratio=0.7, min_unique_word_ratio=0.1):
    if not isinstance(text, str) or not text:
        return False

    # Check 1: Alphabetical character ratio
    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = len(text)
    if total_chars == 0:
        return False
    alpha_ratio = alpha_chars / total_chars
    if alpha_ratio < min_alpha_ratio:
        return False

    # Check 2: Unique word ratio (simple repetition check)
    words = text.split()
    if len(words) < 5: # Require a minimum number of words to check ratio meaningfully
        return True # Assume quality if very short and already passed length filter
    unique_words = set(words)
    unique_word_ratio = len(unique_words) / len(words)
    if unique_word_ratio < min_unique_word_ratio:
        return False

    return True

def generate_shingles(text, k=3):
    if not isinstance(text, str) or not text:
        return []

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove non-alphanumeric characters and replace with spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3. Split the processed text into a list of words
    words = text.split()

    # 4. Generate k-shingles
    shingles = []
    if len(words) < k:
        return [] # Not enough words to form a shingle of length k

    for i in range(len(words) - k + 1):
        shingle = tuple(words[i:i+k])
        shingles.append(shingle)

    return shingles