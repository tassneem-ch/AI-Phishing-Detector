"""
=============================================================================
MAIN PIPELINE — AI-Powered Phishing Detector
=============================================================================
AI-Powered Phishing Detector | Cybersecurity Project
-----------------------------------------------------------------------------
Runs all 5 components in sequence:

    1. Data Ingestion       → load + label emails
    2. Pre-Processing       → clean email text
    3. Feature Extraction   → extract attack indicators + TF-IDF
    4. Classification       → train + evaluate ML models
    5. Threat Reporting     → generate per-email reports + summary

Usage:
    python main.py                      # full pipeline (requires dataset)
    python main.py --demo               # synthetic demo (no dataset needed)
    python main.py --email "path/to/email.txt"   # classify a single email

=============================================================================
"""

import os
import sys
import argparse

# ─── Argument parsing ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="AI-Powered Phishing Detector — Main Pipeline"
)
parser.add_argument(
    "--demo",
    action="store_true",
    help="Run a quick demo using synthetic emails (no dataset required)"
)
parser.add_argument(
    "--email",
    type=str,
    default=None,
    help="Path to a single .txt email file to classify"
)
args = parser.parse_args()


# ─── Imports ──────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from component1_data_ingestion      import load_dataset, split_dataset
from component2_preprocessing       import preprocess_dataframe
from component3_feature_extraction  import (build_feature_matrices,
                                             extract_hand_crafted_features)
from component4_classification      import (run_classification,
                                             train_random_forest,
                                             predict, THREAT_THRESHOLD)
from component5_reporting           import (generate_reports,
                                             print_security_summary,
                                             save_report,
                                             build_single_report)


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 1 — Classify a Single Email File
# ═════════════════════════════════════════════════════════════════════════════

def classify_single_email(filepath, model, vectorizer, tfidf_features=5000):
    """
    Read a single email file, run it through the full pipeline,
    and print the threat report.
    """
    import pickle
    from component2_preprocessing      import preprocess_email
    from component3_feature_extraction import (extract_hand_crafted_features,
                                                URGENCY_WORDS, THREAT_WORDS,
                                                REWARD_WORDS, extract_urls,
                                                is_suspicious_url,
                                                has_sender_mismatch,
                                                has_replyto_mismatch,
                                                count_keyword_hits)
    from scipy.sparse import hstack, csr_matrix

    print(f"\n[Single Email] Reading: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    clean_text = preprocess_email(raw_text)

    # Build a one-row DataFrame to reuse the feature functions
    row = pd.Series({"text": raw_text, "clean_text": clean_text, "label": -1})

    # Hand-crafted features
    urls = extract_urls(raw_text)
    hc_features = {
        "urgency_score":         count_keyword_hits(clean_text, URGENCY_WORDS),
        "threat_score":          count_keyword_hits(clean_text, THREAT_WORDS),
        "reward_score":          count_keyword_hits(clean_text, REWARD_WORDS),
        "link_count":            len(urls),
        "suspicious_link_flag":  int(any(is_suspicious_url(u) for u in urls)),
        "sender_mismatch_flag":  has_sender_mismatch(raw_text),
        "replyto_mismatch_flag": has_replyto_mismatch(raw_text),
    }
    hc_array = np.array(list(hc_features.values()), dtype=np.float32).reshape(1, -1)

    # TF-IDF
    tfidf_array = vectorizer.transform([clean_text])

    # Combined feature vector
    X_single = hstack([csr_matrix(hc_array), tfidf_array])

    # Predict
    proba, pred = predict(model, X_single, THREAT_THRESHOLD)
    threat_score = float(proba[0])
    verdict      = int(pred[0])

    report = build_single_report(raw_text, clean_text, threat_score, verdict, hc_features)
    print(report)


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 2 — Demo with Synthetic Emails
# ═════════════════════════════════════════════════════════════════════════════

def run_demo():
    """Run a full pipeline demonstration using synthetic email examples."""

    print("\n" + "=" * 60)
    print("  DEMO MODE — Synthetic Email Examples")
    print("=" * 60)

    demo_emails = [
        {
            "text": (
                'From: "PayPal Security Team" <security@paypa1-alert.ru>\n'
                'Reply-To: collect@phishnet.tk\n'
                'Subject: URGENT — Account Suspended Due to Unauthorized Access\n\n'
                'Dear Valued Customer,\n\n'
                'We have detected suspicious activity on your PayPal account. '
                'Your account has been temporarily SUSPENDED. '
                'You must verify your identity immediately or your account '
                'will be permanently terminated within 24 hours.\n\n'
                'Click here to verify: http://paypa1-alert.ru/verify?token=abc123\n\n'
                'Failure to act will result in legal action.\n\nPayPal Security'
            ),
            "clean_text": "detected suspicious activity account suspended verify identity immediately terminated 24 hours legal action",
            "label": 1,
        },
        {
            "text": (
                'From: "Alice Johnson" <alice.johnson@mycompany.com>\n'
                'Subject: Q3 Report — Review Required\n\n'
                'Hi team,\n\nPlease find the Q3 financial report attached for your review. '
                'Kindly send any feedback by end of week. '
                'Let me know if you have questions.\n\nBest regards,\nAlice'
            ),
            "clean_text": "find q3 financial report attached review kindly send feedback end week let know questions",
            "label": 0,
        },
        {
            "text": (
                'From: "Prize Department" <winners@free-rewards.xyz>\n'
                'Subject: Congratulations! You Have Been Selected!\n\n'
                'Dear Lucky Winner,\n\n'
                'Congratulations! You have won a $1,000 Amazon gift card! '
                'This is a limited time exclusive offer. Act now to claim your FREE prize. '
                'Click: http://192.168.1.254/claim-prize\n\nExpires in 24 hours!'
            ),
            "clean_text": "congratulations won amazon gift card limited time exclusive offer act now claim free prize expires 24 hours",
            "label": 1,
        },
        {
            "text": (
                'From: "GitHub Notifications" <noreply@github.com>\n'
                'Subject: [GitHub] A new SSH key was added to your account\n\n'
                'Hi, A new public key was added to your GitHub account.\n\n'
                'If you did not perform this action, please remove the key '
                'from your account settings and review your account security.\n\n'
                'Thanks,\nThe GitHub Team'
            ),
            "clean_text": "new public key added github account perform action remove key account settings review account security",
            "label": 0,
        },
        {
            "text": (
                'From: "IT Support" <it-support@comp4ny-helpdesk.tk>\n'
                'Reply-To: attacker@phish.gq\n'
                'Subject: Your password will expire in 24 hours\n\n'
                'Dear Employee,\n\n'
                'Your network password will expire immediately. '
                'To avoid losing access to your systems, click the link below '
                'and update your credentials now:\n\n'
                'http://comp4ny-helpdesk.tk/reset-password\n\n'
                'Do not share this email. Act now.\n\nIT Department'
            ),
            "clean_text": "network password expire immediately avoid losing access systems click link update credentials now act now",
            "label": 1,
        },
    ]

    df = pd.DataFrame(demo_emails)

    X, _, y, _, vec = build_feature_matrices(df, df, tfidf_features=100)
    rf_model = train_random_forest(X, y)

    print("\n[Pipeline] Generating threat reports ...\n")

    proba_arr, pred_arr = predict(rf_model, X, THREAT_THRESHOLD)

    for i, row in df.iterrows():
        hc = extract_hand_crafted_features(row)
        report = build_single_report(
            row["text"], row["clean_text"],
            float(proba_arr[i]), int(pred_arr[i]), hc
        )
        true_label = "PHISH" if row["label"] == 1 else "HAM"
        print(f"\nEmail #{i + 1}  (True label: {true_label})")
        print(report)

    print_security_summary(y, pred_arr, proba_arr, model_name="Random Forest (demo)")


# ═════════════════════════════════════════════════════════════════════════════
#  MODE 3 — Full Pipeline (with real dataset)
# ═════════════════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Run the complete pipeline from data ingestion to threat reporting."""

    print("\n" + "=" * 60)
    print("  AI-POWERED PHISHING DETECTOR — FULL PIPELINE")
    print("=" * 60)

    # ── Step 1: Data Ingestion ─────────────────────────────────────────────
    print("\n[Step 1/5] Data Ingestion")
    print("-" * 40)
    df = load_dataset()
    train_df_raw, test_df_raw = split_dataset(df)

    # ── Step 2: Pre-Processing ─────────────────────────────────────────────
    print("\n[Step 2/5] Pre-Processing")
    print("-" * 40)
    train_df = preprocess_dataframe(train_df_raw)
    test_df  = preprocess_dataframe(test_df_raw)

    # ── Step 3: Feature Extraction ─────────────────────────────────────────
    print("\n[Step 3/5] Threat Signal Extraction")
    print("-" * 40)
    X_train, X_test, y_train, y_test, vectorizer = build_feature_matrices(
        train_df, test_df, tfidf_features=5000
    )

    # ── Step 4: Classification ─────────────────────────────────────────────
    print("\n[Step 4/5] ML Classification Engine")
    print("-" * 40)
    lr_model, rf_model, results = run_classification(
        X_train, X_test, y_train, y_test, save=True
    )

    # Save vectorizer for single-email inference
    import pickle
    os.makedirs("output/models", exist_ok=True)
    with open("output/models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("  Saved → output/models/vectorizer.pkl")

    # ── Step 5: Threat Reporting ───────────────────────────────────────────
    print("\n[Step 5/5] Threat Reporting")
    print("-" * 40)
    reports, proba, preds = generate_reports(
        rf_model, X_test, test_df, max_display=5
    )
    print_security_summary(y_test, preds, proba, model_name="Random Forest")
    save_report(reports, y_test, preds, proba, model_name="Random Forest")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Output files saved to output/")
    print("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if args.email:
        # Single email classification mode
        import pickle
        vec_path   = "output/models/vectorizer.pkl"
        model_path = "output/models/random_forest.pkl"

        if not os.path.exists(vec_path) or not os.path.exists(model_path):
            print("[ERROR] No trained model found. Run 'python main.py' first to train.")
            sys.exit(1)

        with open(vec_path,   "rb") as f: vectorizer = pickle.load(f)
        with open(model_path, "rb") as f: model      = pickle.load(f)

        classify_single_email(args.email, model, vectorizer)

    elif args.demo:
        run_demo()

    else:
        run_full_pipeline()
