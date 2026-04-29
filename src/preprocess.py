import re

def clean_text(text: str) -> str:
    text = str(text)

    # 1. Normalize whitespace
    text = re.sub(r'\s+', ' ' , text).strip()

    # 2. Fix broken ellipses / weird spacing
    text = re.sub(r'\.{2,}', '...', text)

    # 3. Remove control characters (safety for tokenizer)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    return text