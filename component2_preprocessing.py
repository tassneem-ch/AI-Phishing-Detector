"""
=============================================================================
COMPONENT 2 — Pre-Processing Module
=============================================================================
AI-Powered Phishing Detector | Cybersecurity Project
-----------------------------------------------------------------------------
ROLE IN PIPELINE:
    Sanitizes raw email text before feature extraction.
    Attackers embed malicious content inside HTML, use special characters
    to break keyword filters, and pad emails with filler text.
    This module strips all of that noise to expose real content.

INPUT:
    - A pandas DataFrame with columns ["text", "label"]
      (output of Component 1)

OUTPUT:
    - The same DataFrame with a new column "clean_text"
      containing the sanitized, tokenized email content as a plain string
=============================================================================
"""

import re
import pandas as pd
from bs4 import BeautifulSoup


# ─── Stop-word list ───────────────────────────────────────────────────────────
# Common English words that carry no useful signal for classification.
# We define this manually so no external library (NLTK) is needed.

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "we", "our", "you",
    "your", "he", "she", "they", "their", "i", "my", "me", "us", "not",
    "no", "so", "as", "up", "out", "about", "into", "than", "then",
    "there", "here", "when", "where", "who", "which", "what", "how",
    "all", "any", "each", "more", "also", "just", "over", "after",
    "before", "between", "through", "during", "while", "although",
}


# ─── Core cleaning functions ──────────────────────────────────────────────────

def strip_html(text):
    """
    Remove HTML tags and decode HTML entities.

    Security relevance: phishing emails often embed invisible or misleading
    text inside HTML attributes and CSS — stripping HTML exposes what the
    email actually says, not just what it displays.
    """
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    except Exception:
        return text


def remove_urls(text):
    """
    Replace all URLs with the token '__URL__'.

    We replace rather than delete so that the presence of a URL is still
    detectable as a signal (Component 3 will count and analyze the raw URLs
    separately before this step removes them from the text).
    """
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, " __URL__ ", text)


def remove_email_addresses(text):
    """
    Replace email addresses with '__EMAIL__'.
    Keeps the signal that an email address was present without letting
    specific domains bias the classifier.
    """
    email_pattern = r"[\w.\-+]+@[\w.\-]+\.\w+"
    return re.sub(email_pattern, " __EMAIL__ ", text)


def normalize_text(text):
    """
    Lowercase the text and remove everything that is not a letter,
    digit, or one of our placeholder tokens.
    """
    text = text.lower()
    # Keep letters, digits, spaces, and underscores (for __URL__ tokens)
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stop_words(tokens):
    """
    Remove stop words from a list of tokens.
    Returns a filtered list of meaningful words.
    """
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def preprocess_email(raw_text):
    """
    Full pre-processing pipeline for a single email.

    Steps:
        1. Strip HTML tags
        2. Replace URLs with token
        3. Replace email addresses with token
        4. Lowercase and remove punctuation
        5. Tokenize (split into words)
        6. Remove stop words

    Returns a single clean string (tokens joined by spaces).
    """
    text = strip_html(raw_text)
    text = remove_urls(text)
    text = remove_email_addresses(text)
    text = normalize_text(text)

    tokens = text.split()
    tokens = remove_stop_words(tokens)

    return " ".join(tokens)


# ─── Batch processing ─────────────────────────────────────────────────────────

def preprocess_dataframe(df):
    """
    Apply preprocess_email() to every row in the DataFrame.

    Adds a new column 'clean_text' containing the sanitized email.
    The original 'text' column is kept for reference.

    Args:
        df (pd.DataFrame): must have columns ["text", "label"]

    Returns:
        df (pd.DataFrame): same DataFrame with added "clean_text" column
    """
    print("[Pre-Processing] Cleaning email texts ...")
    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess_email)

    # Report any emails that became empty after cleaning
    empty_count = (df["clean_text"].str.strip() == "").sum()
    if empty_count > 0:
        print(f"[WARNING] {empty_count} emails are empty after cleaning — they will be dropped.")
        df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    print(f"[Pre-Processing] Done. {len(df)} emails ready for feature extraction.")
    return df


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" COMPONENT 2 — Pre-Processing")
    print("=" * 60)

    # Example: load from CSV saved by Component 1
    import os
    if os.path.exists("output/train.csv"):
        train_df = pd.read_csv("output/train.csv")
        train_df = preprocess_dataframe(train_df)

        print("\nBefore cleaning:")
        print(train_df["text"].iloc[0][:300])
        print("\nAfter cleaning:")
        print(train_df["clean_text"].iloc[0][:300])

        train_df.to_csv("output/train_clean.csv", index=False)
        print("\nSaved -> output/train_clean.csv")
    else:
        # Quick standalone test
        sample = """
        <html><body>
        <p>Dear Customer,</p>
        <p><b>URGENT:</b> Your account has been <span style='color:red'>SUSPENDED</span>.</p>
        <p>Click <a href='http://paypa1.com/verify?id=abc'>here</a> to verify immediately!</p>
        </body></html>
        """
        print("\nRaw email:")
        print(sample)
        print("\nCleaned:")
        print(preprocess_email(sample))
