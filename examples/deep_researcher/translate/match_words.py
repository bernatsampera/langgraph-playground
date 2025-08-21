import re

from rapidfuzz import fuzz


def match_words_from_glossary(glossary, text, threshold=80):
    """Fuzzy matching for single words AND multi-word phrases.

    Args:
        glossary: dict like {"registrar": "secretario/a", "global history": "historia universal"}
        text: text to search in
        threshold: similarity threshold (0-100)

    Returns:
        dict of matches: {"found_text": "correct_form"}
    """
    matches = {}
    text_lower = text.lower()

    # Check each glossary term
    for term, correct_form in glossary.items():
        term_words = term.split()

        if len(term_words) == 1:
            # Single word matching
            words = re.findall(r"\b[\w-]+\b", text_lower)
            for word in words:
                similarity = fuzz.ratio(term.lower(), word)
                if similarity >= threshold:
                    matches[term] = correct_form
                    break  # Only match once per term
        else:
            # Multi-word phrase matching using sliding window
            text_words = text_lower.split()
            window_size = len(term_words)

            for i in range(len(text_words) - window_size + 1):
                window = " ".join(text_words[i : i + window_size])
                similarity = fuzz.ratio(term.lower(), window)

                if similarity >= threshold:
                    matches[window] = correct_form
                    break  # Only match once per term

    return matches
