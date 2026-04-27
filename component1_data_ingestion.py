"""
=============================================================================
COMPONENT 1 — Data Ingestion Module
=============================================================================
AI-Powered Phishing Detector | Cybersecurity Project
-----------------------------------------------------------------------------
ROLE IN PIPELINE:
    Loads raw emails from disk, assigns threat labels (Phish=1 / Ham=0),
    and splits the dataset into training and test sets.

    In a real deployment this module would be replaced by a live email feed
    from an SMTP gateway or mail-server API.

INPUTS:
    - A folder of Ham emails  (e.g. from the Enron corpus)
    - A folder of Phish emails (e.g. from the SpamAssassin corpus)

OUTPUT:
    - train_df : labeled DataFrame  (80 % of data)
    - test_df  : labeled DataFrame  (20 % of data)

    Each DataFrame has two columns:
        "text"  — raw email body (string)
        "label" — threat label   (0 = Ham, 1 = Phish)
=============================================================================
"""

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split


# ─── Configuration ────────────────────────────────────────────────────────────

# Paths to the raw email folders.
# Update these to point to wherever you store the Enron / SpamAssassin data.
HAM_FOLDERS   = ["data/easy_ham/easy_ham", "data/hard_ham/hard_ham"]    # legitimate emails
PHISH_FOLDERS = ["data/spam_2/spam_2"]  # phishing / spam emails

# How many emails to load per class.
# We keep them equal (balanced) to avoid class-imbalance bias in the model.
MAX_PER_CLASS = 5000

# 80 % training — 20 % testing (standard split in ML security research)
TEST_SIZE     = 0.20
RANDOM_STATE  = 42   # fixed seed → reproducible results


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_emails_from_folder(folder_path, label, max_count):
    """
    Read plain-text email files from a directory and return a list of dicts.

    Each dict has:
        "text"  — the raw content of the file (string)
        "label" — the threat label passed as argument (int: 0 or 1)

    Files that cannot be decoded (binary attachments, etc.) are skipped.
    """
    records = []

    if not os.path.isdir(folder_path):
        print(f"[WARNING] Folder not found: {folder_path}")
        return records

    files = os.listdir(folder_path)
    random.shuffle(files)          # shuffle so we get a diverse sample
    files = files[:max_count]      # cap at max_count

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:                           # skip empty files
                records.append({"text": text, "label": label})
        except Exception as e:
            print(f"[WARNING] Could not read {filename}: {e}")

    return records


def load_dataset(ham_folders=HAM_FOLDERS,
                 phish_folders=PHISH_FOLDERS,
                 max_per_class=MAX_PER_CLASS):
    """
    Load Ham and Phish emails, combine them into a single DataFrame,
    and shuffle the rows so classes are interleaved.

    Returns:
        df (pd.DataFrame) with columns ["text", "label"]
    """
    print("[1/4] Loading Ham emails ...")
    ham_records = []
    for folder in ham_folders:
        ham_records.extend(load_emails_from_folder(folder, label=0, max_count=max_per_class))
    random.shuffle(ham_records)
    ham_records = ham_records[:max_per_class]

    print("[2/4] Loading Phish emails ...")
    phish_records = []
    for folder in phish_folders:
        phish_records.extend(load_emails_from_folder(folder, label=1, max_count=max_per_class))
    random.shuffle(phish_records)
    phish_records = phish_records[:max_per_class]

    if not ham_records and not phish_records:
        raise FileNotFoundError(
            "No emails found. Make sure the data/ folders exist and contain .txt files.\n"
            "See README.md for download instructions."
        )

    all_records = ham_records + phish_records
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

    ham_count   = (df["label"] == 0).sum()
    phish_count = (df["label"] == 1).sum()
    print(f"[3/4] Dataset loaded: {len(df)} emails total "
          f"({ham_count} Ham, {phish_count} Phish)")

    return df


def split_dataset(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split the dataset into training and test sets using stratified sampling.

    Stratified = we preserve the Ham/Phish ratio in both sets,
    which is important for reliable security evaluation metrics.

    Returns:
        train_df (pd.DataFrame)
        test_df  (pd.DataFrame)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]   # keep class ratio equal in both sets
    )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"[4/4] Split complete -> "
          f"Training: {len(train_df)} emails | Test: {len(test_df)} emails")

    return train_df, test_df


# ─── Demo (runs only when this file is executed directly) ────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" COMPONENT 1 — Data Ingestion")
    print("=" * 60)

    df = load_dataset()
    train_df, test_df = split_dataset(df)

    print("\nSample rows from training set:")
    print(train_df.head(3).to_string())

    # Save to CSV so other components can load quickly during development
    os.makedirs("output", exist_ok=True)
    train_df.to_csv("output/train.csv", index=False)
    test_df.to_csv("output/test.csv",   index=False)
    print("\nSaved -> output/train.csv and output/test.csv")
