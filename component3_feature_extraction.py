"""
=============================================================================
COMPONENT 3 — Threat Signal Extraction Module
=============================================================================
AI-Powered Phishing Detector | Cybersecurity Project
-----------------------------------------------------------------------------
ROLE IN PIPELINE:
    Transforms cleaned emails into a numeric feature matrix that the ML
    classifier can process. Each feature maps to a documented phishing
    attack technique.

    Two types of features are produced:
        A) Hand-crafted attack indicators (6 features per email)
        B) TF-IDF vectors  (statistical word importance across the corpus)

    These are combined into a single feature matrix passed to Component 4.

INPUTS:
    - train_df : pre-processed DataFrame (from Component 2)
    - test_df  : pre-processed DataFrame (from Component 2)
    Both must have "clean_text" and "text" columns.

OUTPUTS:
    - X_train, X_test : numeric feature matrices (NumPy / sparse arrays)
    - y_train, y_test : threat label arrays (0 = Ham, 1 = Phish)
    - vectorizer      : fitted TF-IDF vectorizer (saved for inference)
=============================================================================
"""

import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── Attack Indicator Lexicons ────────────────────────────────────────────────
# Words and phrases associated with phishing social-engineering techniques.
# Source: MITRE ATT&CK T1566 (Phishing), common threat intelligence reports.

URGENCY_WORDS = [
    "urgent", "immediately", "right away", "act now", "expires",
    "expiring", "limited time", "last chance", "final notice",
    "within 24 hours", "within 48 hours", "as soon as possible",
    "asap", "don't delay", "no later than", "deadline", "time sensitive",
    "respond immediately", "respond now",
]

THREAT_WORDS = [
    "suspended", "terminated", "blocked", "compromised", "unauthorized",
    "illegal", "fraudulent", "suspicious activity", "security alert",
    "security breach", "account locked", "access denied", "legal action",
    "law enforcement", "police", "arrested", "penalty", "fine",
]

REWARD_WORDS = [
    "congratulations", "you have won", "you've been selected",
    "free gift", "prize", "reward", "bonus", "claim now",
    "claim your", "lucky winner", "exclusive offer", "special offer",
    "100% free", "no cost", "cash prize",
]

# Suspicious top-level domains frequently used in phishing infrastructure
SUSPICIOUS_TLDS = [
    ".ru", ".cn", ".tk", ".xyz", ".top", ".click", ".link",
    ".download", ".loan", ".work", ".party", ".gq", ".ml",
]


# ─── Hand-crafted Feature Extraction ─────────────────────────────────────────

def count_keyword_hits(text, keyword_list):
    """
    Count how many keywords from the list appear in the text.
    Case-insensitive. Returns an integer count.
    """
    text_lower = text.lower()
    return sum(1 for kw in keyword_list if kw in text_lower)


def extract_urls(raw_text):
    """
    Extract all URLs from the raw (uncleaned) email text.
    Returns a list of URL strings.
    """
    pattern = r"https?://[^\s\"'<>]+"
    return re.findall(pattern, raw_text)


def is_suspicious_url(url):
    """
    Checks whether a URL matches common phishing domain patterns:
        - Uses a raw IP address instead of a domain name
        - Contains a suspicious TLD
        - Uses a lookalike brand name with numbers/hyphens (e.g. paypa1, amazon-secure)

    Returns True if the URL looks suspicious.
    """
    # IP-based URL: attacker avoids registering a domain
    ip_pattern = r"https?://\d{1,3}(\.\d{1,3}){3}"
    if re.match(ip_pattern, url):
        return True

    # Suspicious TLD
    for tld in SUSPICIOUS_TLDS:
        if tld in url.lower():
            return True

    # Digits inside what looks like a brand name (paypa1, amaz0n, g00gle)
    domain_part = re.sub(r"https?://", "", url).split("/")[0]
    if re.search(r"[a-z]+\d+[a-z]+", domain_part):
        return True

    return False


def has_sender_mismatch(raw_text):
    """
    Detects sender spoofing: checks whether the display name in the
    'From' header and the actual email domain are inconsistent.

    Example attack: From: "PayPal Support <no-reply@random123.ru>"

    Returns 1 if a mismatch is detected, 0 otherwise.
    """
    # Look for a From header
    from_match = re.search(r"(?i)^from:(.+)$", raw_text, re.MULTILINE)
    if not from_match:
        return 0

    from_line = from_match.group(1)

    # Extract the display name (before the <) and the email address (inside <>)
    name_match  = re.search(r'"?([^"<]+)"?\s*<', from_line)
    addr_match  = re.search(r"<([^>]+)>", from_line)

    if not name_match or not addr_match:
        return 0

    display_name = name_match.group(1).strip().lower()
    email_domain = addr_match.group(1).split("@")[-1].strip().lower() if "@" in addr_match.group(1) else ""

    # Known brands that should have matching domains
    known_brands = {
        "paypal": "paypal.com",
        "amazon": "amazon.com",
        "apple":  "apple.com",
        "google": "google.com",
        "microsoft": "microsoft.com",
        "facebook": "facebook.com",
        "netflix": "netflix.com",
        "ebay": "ebay.com",
        "dhl": "dhl.com",
        "fedex": "fedex.com",
    }

    for brand, expected_domain in known_brands.items():
        if brand in display_name and expected_domain not in email_domain:
            return 1   # Mismatch detected — likely spoofing

    return 0


def has_replyto_mismatch(raw_text):
    """
    Detects reply hijacking: checks whether the Reply-To address
    differs from the From address, which silently redirects replies
    to the attacker.

    Returns 1 if a mismatch is found, 0 otherwise.
    """
    from_match    = re.search(r"(?i)^from:\s*.*?([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]+)", raw_text, re.MULTILINE)
    replyto_match = re.search(r"(?i)^reply-to:\s*.*?([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]+)", raw_text, re.MULTILINE)

    if from_match and replyto_match:
        from_addr    = from_match.group(1).strip().lower()
        replyto_addr = replyto_match.group(1).strip().lower()
        if from_addr != replyto_addr:
            return 1

    return 0


def extract_hand_crafted_features(row):
    """
    Extract all 6 hand-crafted attack indicators for a single email.

    Args:
        row: a pandas Series with "text" (raw) and "clean_text" (cleaned)

    Returns:
        dict with 6 numeric features
    """
    raw_text   = str(row.get("text", ""))
    clean_text = str(row.get("clean_text", ""))

    urls = extract_urls(raw_text)
    suspicious_link = int(any(is_suspicious_url(u) for u in urls))

    return {
        "urgency_score":        count_keyword_hits(clean_text, URGENCY_WORDS),
        "threat_score":         count_keyword_hits(clean_text, THREAT_WORDS),
        "reward_score":         count_keyword_hits(clean_text, REWARD_WORDS),
        "link_count":           len(urls),
        "suspicious_link_flag": suspicious_link,
        "sender_mismatch_flag": has_sender_mismatch(raw_text),
        "replyto_mismatch_flag":has_replyto_mismatch(raw_text),
    }


def build_hand_crafted_matrix(df):
    """
    Apply extract_hand_crafted_features() to every row in the DataFrame.
    Returns a dense NumPy array of shape (n_emails, 7).
    """
    print("[Feature Extraction] Computing hand-crafted attack indicators ...")
    features = df.apply(extract_hand_crafted_features, axis=1)
    matrix   = pd.DataFrame(list(features)).values.astype(np.float32)
    print(f"    Hand-crafted matrix shape: {matrix.shape}")
    return matrix


# ─── TF-IDF Vectorization ─────────────────────────────────────────────────────

def build_tfidf_features(train_texts, test_texts, max_features=5000):
    """
    Fit a TF-IDF vectorizer on the training set and transform both sets.

    TF-IDF (Term Frequency–Inverse Document Frequency) gives high scores
    to words that are frequent in a specific email but rare across the
    whole corpus — these are the most discriminative words for classification.

    The vectorizer is fitted ONLY on training data to prevent data leakage
    (a common security evaluation mistake that inflates detection metrics).

    Args:
        train_texts  : list/Series of clean email strings (training set)
        test_texts   : list/Series of clean email strings (test set)
        max_features : vocabulary size cap (top N most informative words)

    Returns:
        X_train_tfidf : sparse matrix
        X_test_tfidf  : sparse matrix
        vectorizer    : fitted TfidfVectorizer (needed for live inference)
    """
    print(f"[Feature Extraction] Fitting TF-IDF vectorizer (top {max_features} features) ...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # unigrams + bigrams (e.g. "verify account")
        min_df=2,                # ignore words that appear in fewer than 2 emails
        sublinear_tf=True,       # apply log normalization to term frequencies
    )

    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf  = vectorizer.transform(test_texts)

    print(f"    TF-IDF matrix shape: {X_train_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectorizer


# ─── Combined Feature Matrix ──────────────────────────────────────────────────

def build_feature_matrices(train_df, test_df, tfidf_features=5000):
    """
    Build the complete feature matrices for training and test sets by
    combining hand-crafted attack indicators with TF-IDF vectors.

    Args:
        train_df        : pre-processed training DataFrame
        test_df         : pre-processed test DataFrame
        tfidf_features  : number of TF-IDF vocabulary features

    Returns:
        X_train : combined feature matrix for training
        X_test  : combined feature matrix for testing
        y_train : threat labels for training (0 or 1)
        y_test  : threat labels for testing  (0 or 1)
        vectorizer : fitted TF-IDF vectorizer
    """
    print("\n[Feature Extraction] Building feature matrices ...")

    # 1) Hand-crafted attack indicators
    hc_train = build_hand_crafted_matrix(train_df)
    hc_test  = build_hand_crafted_matrix(test_df)

    # Convert to sparse for efficient concatenation with TF-IDF
    hc_train_sparse = csr_matrix(hc_train)
    hc_test_sparse  = csr_matrix(hc_test)

    # 2) TF-IDF vectors
    tfidf_train, tfidf_test, vectorizer = build_tfidf_features(
        train_df["clean_text"],
        test_df["clean_text"],
        max_features=tfidf_features
    )

    # 3) Combine both feature types side-by-side
    X_train = hstack([hc_train_sparse, tfidf_train])
    X_test  = hstack([hc_test_sparse,  tfidf_test])

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    print(f"\n[Feature Extraction] Final feature matrix shape:")
    print(f"    X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"    Features: 7 hand-crafted attack indicators + {tfidf_features} TF-IDF features")

    return X_train, X_test, y_train, y_test, vectorizer


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" COMPONENT 3 — Threat Signal Extraction")
    print("=" * 60)

    import os
    if os.path.exists("output/train_clean.csv") and os.path.exists("output/test.csv"):
        from component2_preprocessing import preprocess_dataframe

        train_df = pd.read_csv("output/train_clean.csv")
        test_df  = pd.read_csv("output/test.csv")
        test_df  = preprocess_dataframe(test_df)

        X_train, X_test, y_train, y_test, vec = build_feature_matrices(train_df, test_df)
        print(f"\nTraining set: {X_train.shape[0]} emails, {X_train.shape[1]} features each")
        print(f"Test set    : {X_test.shape[0]} emails, {X_test.shape[1]} features each")
    else:
        # Standalone demo on synthetic examples
        sample_phish = pd.DataFrame([{
            "text": 'From: "PayPal Support" <no-reply@random-domain.ru>\n'
                    'Reply-To: attacker@evil.com\n\n'
                    'URGENT: Your account has been suspended due to unauthorized access. '
                    'Click http://paypa1.com/verify immediately to restore access or face legal action.',
            "clean_text": "urgent account suspended unauthorized access click immediately restore legal action",
            "label": 1
        }])
        sample_ham = pd.DataFrame([{
            "text": 'From: "Alice Smith" <alice@company.com>\n\n'
                    'Hi Bob, just sending the meeting notes from today. Let me know if you have questions.',
            "clean_text": "hi sending meeting notes today let know questions",
            "label": 0
        }])

        train_df = pd.concat([sample_phish, sample_ham], ignore_index=True)
        test_df  = train_df.copy()

        X_train, X_test, y_train, y_test, vec = build_feature_matrices(train_df, test_df, tfidf_features=50)

        print("\nHand-crafted features for the phishing sample:")
        feats = extract_hand_crafted_features(sample_phish.iloc[0])
        for k, v in feats.items():
            print(f"  {k}: {v}")
