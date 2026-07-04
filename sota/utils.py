import re

def clean_text(examples):
    # Remove multiple spaces and newlines, replace with a single space
    cleaned_texts = [re.sub(r'\s+', ' ', text).strip() for text in examples['text']]
    # Filter out empty strings that might result from cleaning
    return {'text': [text for text in cleaned_texts if text]}