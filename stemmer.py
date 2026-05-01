import re
from nltk.tokenize import word_tokenize

# =========================
# Configuration
# =========================

VOWELS = set("aeiou")
MIN_STEM_LENGTH = 3

# Ordered longest → shortest
SUFFIX_RULES = [
    "ootii", "oota", "oti",
    "uu", "e", "a",
    "aa", "ii",
]

# =========================
# Validation rules
# =========================

def is_valid_stem(stem: str) -> bool:
    """
    Conservative validation to avoid over-stemming.
    """
    if len(stem) < MIN_STEM_LENGTH:
        return False

    # Afaan Oromo stems should contain at least one vowel
    if not any(vowel in stem for vowel in VOWELS):
        return False

    return True


# =========================
# Core stemming function
# =========================

def stem_word(word: str) -> str:
    """
    Applies ONE suffix removal according to ordered rules.
    """
    for suffix in SUFFIX_RULES:
        if word.endswith(suffix):
            candidate = word[:-len(suffix)]
            if is_valid_stem(candidate):
                return candidate
    return word


# =========================
# Full stemming pipeline
# =========================

def afaan_oromo_stem(text: str):
    """
    Tokenizes and stems Afaan Oromo text for Information Retrieval.
    """
    # Note: Ensure you have run nltk.download('punkt')
    tokens = word_tokenize(text.lower())
    stems = []

    for token in tokens:
        # keep only words (no numbers or punctuation)
        if re.fullmatch(r"[a-z']+", token):
            stems.append(stem_word(token))
    
    return stems


if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    stems = afaan_oromo_stem(text)

    print("Stemmed output:")
    print(stems)
