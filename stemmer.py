import re
from nltk.tokenize import word_tokenize

# =====================================================
# PHONOLOGICAL & VALIDATION CONSTANTS
# =====================================================

VOWELS = set("aeiou")
MIN_STEM_LENGTH = 3
MAX_RECURSION = 4   # safety bound against over-stripping

# =====================================================
# MORPHOLOGICAL SUFFIX GROUPS
# =====================================================

# Plural markers
PLURAL_SUFFIXES = [
    "ootii", "oota", "oti", "wwan"
]

# Case markers
CASE_SUFFIXES = [
    "tti", "f", "irraa", "dhaaf"
]

# Possessive markers
POSSESSIVE_SUFFIXES = [
    "koo", "kee", "isaa", "ishee", "keenya", "isaanii"
]

# Verb inflectional endings
VERB_SUFFIXES = [
    "achuun", "achuun",  # gerundive variants
    "achuun", "achuu",
    "anii", "atte", "ata", "an",
    "te", "ne", "ta", "ti",
    "uu", "e", "a"
]

# Derivational morphology
DERIVATIONAL_SUFFIXES = [
    "ummaa", "eenya", "annoo", "suu", "sa", "nya"
]

# =====================================================
# VALIDATION FUNCTIONS
# =====================================================

def is_valid_stem(stem: str) -> bool:
    """
    Strong stem validation to prevent errors.
    """
    if len(stem) < MIN_STEM_LENGTH:
        return False

    if not any(v in stem for v in VOWELS):
        return False

    if re.search(r"[^a-z']", stem):
        return False

    return True


# =====================================================
# POS HEURISTICS
# =====================================================

def guess_pos(word: str) -> str:
    """
    Lightweight POS heuristics.
    Full stemmers REQUIRE POS separation.
    """
    if any(word.endswith(s) for s in VERB_SUFFIXES):
        return "VERB"

    if any(word.endswith(s) for s in PLURAL_SUFFIXES + CASE_SUFFIXES):
        return "NOUN"

    return "UNKNOWN"


# =====================================================
# RECURSIVE STRIPPING ENGINE
# =====================================================

def recursive_strip(word: str, suffixes: list) -> str:
    """
    Controlled recursive stripping with rollback protection.
    """
    stem = word
    steps = 0

    while steps < MAX_RECURSION:
        stripped = False
        for suf in suffixes:
            if stem.endswith(suf):
                candidate = stem[:-len(suf)]
                if is_valid_stem(candidate):
                    stem = candidate
                    stripped = True
                    break
        if not stripped:
            break
        steps += 1

    return stem


# =====================================================
# FULL STEMMER CORE
# =====================================================

def full_stem_word(word: str) -> str:
    """
    Full morphological normalization for Afaan Oromo.
    """
    original = word
    pos = guess_pos(word)

    stem = word

    if pos == "NOUN":
        stem = recursive_strip(stem, CASE_SUFFIXES)
        stem = recursive_strip(stem, POSSESSIVE_SUFFIXES)
        stem = recursive_strip(stem, PLURAL_SUFFIXES)
        stem = recursive_strip(stem, DERIVATIONAL_SUFFIXES)

    elif pos == "VERB":
        stem = recursive_strip(stem, VERB_SUFFIXES)
        stem = recursive_strip(stem, DERIVATIONAL_SUFFIXES)

    else:
        stem = recursive_strip(stem, DERIVATIONAL_SUFFIXES)

    # rollback safety
    if is_valid_stem(stem):
        return stem

    return original


# =====================================================
# PIPELINE FUNCTION
# =====================================================

def afaan_oromo_full_stem(text: str):
    """
    Full stemming pipeline.
    """
    tokens = word_tokenize(text.lower())
    stems = []

    for token in tokens:
        if re.fullmatch(r"[a-z']+", token):
            stems.append(full_stem_word(token))

    return stems


# =====================================================
# Testing and demonstration
# =====================================================

if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    stems = afaan_oromo_full_stem(text)

    print("Stemmed output:")
    print(stems)
