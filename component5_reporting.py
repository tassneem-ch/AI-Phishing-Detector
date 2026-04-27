"""
=============================================================================
COMPONENT 5 — Threat Reporting Module
=============================================================================
AI-Powered Phishing Detector | Cybersecurity Project
-----------------------------------------------------------------------------
ROLE IN PIPELINE:
    Translates the classifier's raw output into human-readable threat
    reports. For every email analyzed it shows:
        - The verdict        : PHISH ⚠  or  HAM ✓
        - The threat score   : probability (e.g. 91 %)
        - Triggered signals  : which specific attack indicators fired
        - Overall statistics : Accuracy, Precision, Recall, F1 on the
                               full test set, plus a confusion matrix plot

    In a real security deployment this module would feed into a SIEM
    dashboard or send alerts to an analyst queue.

INPUTS:
    - Trained model        (from Component 4)
    - Feature matrices     (from Component 3)
    - Raw / cleaned emails (for context in the report)
    - y_true               (ground-truth labels, for evaluation)

OUTPUTS:
    - Printed threat reports per email
    - output/threat_report.txt  — full text report saved to disk
    - output/confusion_matrix.png — visual evaluation chart
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works without a screen
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score)

from component4_classification import predict, THREAT_THRESHOLD


# ─── Threat Score → Risk Level ────────────────────────────────────────────────

def score_to_risk(score):
    """Map a probability score to a human-readable risk level."""
    if score >= 0.85:
        return "CRITICAL"
    elif score >= 0.65:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    else:
        return "LOW / SAFE"


# ─── Single Email Threat Report ───────────────────────────────────────────────

def build_single_report(email_text, clean_text, threat_score,
                        verdict, hand_crafted_features):
    """
    Build a structured threat report string for one email.

    Args:
        email_text           : raw email (for subject / sender display)
        clean_text           : pre-processed email text
        threat_score         : float 0.0–1.0 from the classifier
        verdict              : 1 = Phish, 0 = Ham
        hand_crafted_features: dict of the 7 attack indicators for this email

    Returns:
        A formatted multi-line string report.
    """
    lines = []
    sep   = "-" * 56

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(sep)
    if verdict == 1:
        lines.append("  [!] VERDICT: PHISH  (Malicious Email Detected)")
    else:
        lines.append("  [OK] VERDICT: HAM   (Email Appears Legitimate)")

    lines.append(f"  Threat Score : {threat_score * 100:.1f}%")
    lines.append(f"  Risk Level   : {score_to_risk(threat_score)}")
    lines.append(sep)

    # ── Email preview ─────────────────────────────────────────────────────────
    # Extract Subject and From from raw text headers if available
    subject = "N/A"
    sender  = "N/A"
    for line in email_text.splitlines()[:20]:
        if line.lower().startswith("subject:"):
            subject = line[8:].strip()
        if line.lower().startswith("from:"):
            sender = line[5:].strip()

    lines.append(f"  From    : {sender[:70]}")
    lines.append(f"  Subject : {subject[:70]}")
    lines.append(sep)

    # ── Attack Indicators ─────────────────────────────────────────────────────
    lines.append("  ATTACK INDICATORS DETECTED:")
    lines.append("")

    hc = hand_crafted_features

    # Urgency
    u = int(hc.get("urgency_score", 0))
    flag = "[!]" if u > 0 else "[OK]"
    lines.append(f"  {flag}  Urgency keywords       : {u} hit(s)")

    # Threat language
    t = int(hc.get("threat_score", 0))
    flag = "[!]" if t > 0 else "[OK]"
    lines.append(f"  {flag}  Threat/fear language   : {t} hit(s)")

    # Reward language
    r = int(hc.get("reward_score", 0))
    flag = "[!]" if r > 0 else "[OK]"
    lines.append(f"  {flag}  Reward/greed language  : {r} hit(s)")

    # Link count
    lc = int(hc.get("link_count", 0))
    flag = "[!]" if lc > 3 else "[OK]"
    lines.append(f"  {flag}  URLs found             : {lc}")

    # Suspicious link
    sl = int(hc.get("suspicious_link_flag", 0))
    flag = "[!]" if sl else "[OK]"
    lines.append(f"  {flag}  Suspicious URL detected: {'YES — possible domain spoofing / typosquatting' if sl else 'No'}")

    # Sender mismatch
    sm = int(hc.get("sender_mismatch_flag", 0))
    flag = "[!]" if sm else "[OK]"
    lines.append(f"  {flag}  Sender spoofing        : {'YES — display name vs domain mismatch' if sm else 'No'}")

    # Reply-To mismatch
    rm = int(hc.get("replyto_mismatch_flag", 0))
    flag = "[!]" if rm else "[OK]"
    lines.append(f"  {flag}  Reply-To hijacking     : {'YES — replies redirected to different address' if rm else 'No'}")

    lines.append(sep)

    # ── Recommended Action ────────────────────────────────────────────────────
    if verdict == 1:
        lines.append("  RECOMMENDED ACTION:")
        if threat_score >= 0.85:
            lines.append("  -> Quarantine immediately. Do NOT click any links.")
            lines.append("  -> Report to your IT/Security team.")
        elif threat_score >= 0.65:
            lines.append("  -> Flag for manual review. Avoid clicking links.")
        else:
            lines.append("  -> Monitor. Treat with caution.")
    else:
        lines.append("  RECOMMENDED ACTION: No threat detected. Email delivered.")

    lines.append(sep)
    return "\n".join(lines)


# ─── Batch Reporting ──────────────────────────────────────────────────────────

def generate_reports(model, X_test, test_df,
                     threshold=THREAT_THRESHOLD,
                     max_display=10):
    """
    Generate threat reports for a batch of emails.

    Args:
        model       : trained classifier (from Component 4)
        X_test      : feature matrix for the test set
        test_df     : DataFrame with "text", "clean_text", "label" columns
        threshold   : classification threshold
        max_display : how many individual reports to print (full list saved)

    Returns:
        all_reports : list of report strings
        proba_array : threat scores for all emails
        pred_array  : verdict array (0 or 1)
    """
    from component3_feature_extraction import extract_hand_crafted_features

    proba_array, pred_array = predict(model, X_test, threshold)

    all_reports = []

    for i in range(len(test_df)):
        row          = test_df.iloc[i]
        email_text   = str(row.get("text",       ""))
        clean_text   = str(row.get("clean_text", ""))
        threat_score = float(proba_array[i])
        verdict      = int(pred_array[i])
        hc_features  = extract_hand_crafted_features(row)

        report = build_single_report(
            email_text, clean_text, threat_score, verdict, hc_features
        )
        all_reports.append(report)

        if i < max_display:
            print(f"\nEmail #{i + 1}  (True label: {'PHISH' if row['label'] == 1 else 'HAM'})")
            print(report)

    if len(test_df) > max_display:
        print(f"\n... {len(test_df) - max_display} more emails processed "
              f"(see output/threat_report.txt for full results)")

    return all_reports, proba_array, pred_array


# ─── Overall Security Metrics ─────────────────────────────────────────────────

def print_security_summary(y_true, y_pred, y_proba, model_name="Best Model"):
    """
    Print the overall detection performance of the model on the test set.
    Also saves a confusion matrix chart.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 56)
    print(f"  OVERALL DETECTION PERFORMANCE — {model_name}")
    print("=" * 56)
    print(f"  Total emails analyzed  : {len(y_true)}")
    print(f"  Phishing emails        : {int(y_true.sum())}")
    print(f"  Legitimate emails      : {int((y_true == 0).sum())}")
    print()
    print(f"  Accuracy               : {acc * 100:.2f}%")
    print(f"  Precision              : {prec * 100:.2f}%")
    print(f"  Recall (Detection Rate): {rec * 100:.2f}%")
    print(f"  F1-Score               : {f1 * 100:.2f}%")

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print()
        print(f"  Attacks correctly detected (TP)   : {tp}")
        print(f"  Attacks missed         (FN) [!]   : {fn}  <- security failures")
        print(f"  False alarms           (FP)        : {fp}")
        print(f"  Legitimate emails correctly passed : {tn}")
    print("=" * 56)

    # Save confusion matrix chart
    _plot_confusion_matrix(cm)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _plot_confusion_matrix(cm):
    """Save a styled confusion matrix heatmap to disk."""
    os.makedirs("output", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = [["#2ecc71", "#e74c3c"],
              ["#e67e22", "#2980b9"]]
    labels = [["True Negative\n(Ham correctly passed)",
               "False Positive\n(Ham flagged as Phish)"],
              ["False Negative\n(Phish missed ⚠)",
               "True Positive\n(Phish detected ✓)"]]

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                        color=colors[i][j], alpha=0.75))
            val = cm[i, j] if cm.shape == (2, 2) else 0
            ax.text(j + 0.5, 1 - i + 0.6, str(val),
                    ha="center", va="center",
                    fontsize=22, fontweight="bold", color="white")
            ax.text(j + 0.5, 1 - i + 0.25, labels[i][j],
                    ha="center", va="center",
                    fontsize=8, color="white")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted Ham", "Predicted Phish"], fontsize=11)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual Phish", "Actual Ham"], fontsize=11)
    ax.set_title("Confusion Matrix — Phishing Detector", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    path = "output/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Confusion matrix saved -> {path}")


# ─── Save Full Report to File ─────────────────────────────────────────────────

def save_report(all_reports, y_true, y_pred, y_proba, model_name):
    """Write all individual email reports and the summary to a text file."""
    os.makedirs("output", exist_ok=True)
    path = "output/threat_report.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("AI-POWERED PHISHING DETECTOR — THREAT REPORT\n")
        f.write("=" * 56 + "\n\n")

        for idx, report in enumerate(all_reports):
            f.write(f"Email #{idx + 1}\n")
            f.write(report + "\n\n")

        # Summary section
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        f.write("\n" + "=" * 56 + "\n")
        f.write(f"SUMMARY — {model_name}\n")
        f.write("=" * 56 + "\n")
        f.write(f"Total emails : {len(y_true)}\n")
        f.write(f"Accuracy     : {acc * 100:.2f}%\n")
        f.write(f"Precision    : {prec * 100:.2f}%\n")
        f.write(f"Recall       : {rec * 100:.2f}%\n")
        f.write(f"F1-Score     : {f1 * 100:.2f}%\n")

    print(f"  Full threat report saved -> {path}")


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" COMPONENT 5 — Threat Reporting Module")
    print("=" * 60)

    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from component2_preprocessing      import preprocess_dataframe
    from component3_feature_extraction import build_feature_matrices
    from component4_classification     import (train_random_forest,
                                               load_model, save_model)

    if os.path.exists("output/train_clean.csv") and os.path.exists("output/test.csv"):
        train_df = pd.read_csv("output/train_clean.csv")
        test_df  = preprocess_dataframe(pd.read_csv("output/test.csv"))

        X_train, X_test, y_train, y_test, _ = build_feature_matrices(train_df, test_df)

        # Load saved model if available, otherwise train fresh
        model_path = "output/models/random_forest.pkl"
        if os.path.exists(model_path):
            print("[Reporting] Loading saved Random Forest model ...")
            rf_model = load_model(model_path)
        else:
            print("[Reporting] No saved model found — training now ...")
            rf_model = train_random_forest(X_train, y_train)

        # Generate reports
        reports, proba, preds = generate_reports(
            rf_model, X_test, test_df, max_display=5
        )

        # Overall summary + confusion matrix
        print_security_summary(y_test, preds, proba, model_name="Random Forest")

        # Save everything
        save_report(reports, y_test, preds, proba, model_name="Random Forest")

    else:
        # Synthetic demo — no dataset needed
        print("[INFO] No dataset found — running synthetic demo ...\n")

        # Create fake emails for demonstration
        demo_emails = [
            {
                "text": (
                    'From: "PayPal Security" <alert@paypa1-secure.ru>\n'
                    'Reply-To: harvest@evil.com\n'
                    'Subject: URGENT: Your account has been suspended\n\n'
                    'Dear customer, we detected unauthorized access. '
                    'Your account will be terminated in 24 hours. '
                    'Click http://paypa1-secure.ru/verify immediately to avoid legal action.'
                ),
                "clean_text": "detected unauthorized access account terminated 24 hours click immediately avoid legal action",
                "label": 1,
            },
            {
                "text": (
                    'From: "Alice Johnson" <alice.johnson@company.com>\n'
                    'Subject: Meeting notes from today\n\n'
                    'Hi team, please find the meeting notes attached. '
                    'Let me know if you have any questions. Thanks, Alice.'
                ),
                "clean_text": "find meeting notes attached let know questions thanks",
                "label": 0,
            },
            {
                "text": (
                    'From: "Prize Center" <winner@free-gifts.tk>\n'
                    'Subject: Congratulations! You have won!\n\n'
                    'You have been selected as our lucky winner! '
                    'Claim your FREE prize now! Limited time offer — act now!'
                ),
                "clean_text": "selected lucky winner claim free prize now limited time offer act now",
                "label": 1,
            },
        ]

        demo_df = pd.DataFrame(demo_emails)

        from component3_feature_extraction import (build_feature_matrices,
                                                    extract_hand_crafted_features)
        from component4_classification     import train_random_forest
        from scipy.sparse                  import csr_matrix
        import numpy as np

        # Build minimal features for the demo
        X, _, y, _, _ = build_feature_matrices(demo_df, demo_df, tfidf_features=50)
        rf_model = train_random_forest(X, y)

        proba_arr, pred_arr = predict(rf_model, X, threshold=THREAT_THRESHOLD)

        print()
        for i, row in demo_df.iterrows():
            hc = extract_hand_crafted_features(row)
            report = build_single_report(
                row["text"], row["clean_text"],
                float(proba_arr[i]), int(pred_arr[i]), hc
            )
            print(f"\nEmail #{i + 1}  (True label: {'PHISH' if row['label'] == 1 else 'HAM'})")
            print(report)

        print_security_summary(y, pred_arr, proba_arr, model_name="Random Forest (demo)")
