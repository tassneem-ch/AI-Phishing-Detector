# AI-Powered Phishing Detector

A complete, end-to-end Machine Learning pipeline for detecting and classifying phishing emails. This cybersecurity project uses both hand-crafted threat indicators and TF-IDF statistical features to train models (Logistic Regression and Random Forest) to identify malicious emails.

## 🚀 Pipeline Components

The system is highly modular and broken down into 5 sequential components:

1. **Data Ingestion (`component1_data_ingestion.py`)**  
   Loads raw legitimate (Ham) and malicious (Phish) emails from disk, assigns threat labels, and performs a stratified split into training and test datasets.
2. **Pre-Processing (`component2_preprocessing.py`)**  
   Sanitizes raw email text by stripping HTML, tokenizing URLs and email addresses, and removing English stop words to expose the true textual content.
3. **Threat Signal Extraction (`component3_feature_extraction.py`)**  
   Transforms cleaned emails into a numeric feature matrix using:
   - **Hand-crafted attack indicators:** Checks for urgency words, threat language, reward/greed language, suspicious TLDs, sender spoofing, and reply-to hijacking.
   - **TF-IDF vectors:** Extracts the statistical importance of words across the email corpus.
4. **ML Classification Engine (`component4_classification.py`)**  
   Trains and compares Logistic Regression and Random Forest models on the extracted features. Random Forest serves as the main classifier, outputting threat probabilities.
5. **Threat Reporting (`component5_reporting.py`)**  
   Translates the classifier's raw output into human-readable threat reports. Generates per-email verdicts (PHISH ⚠ or HAM ✓), lists triggered attack indicators, and outputs overall security evaluation metrics (Precision, Recall, F1, Confusion Matrix).

## 🛠️ Usage

The project provides an orchestrator script (`main.py`) that supports three different execution modes.

### 1. Demo Mode (No Dataset Required)
Run a quick, full-pipeline demonstration using built-in synthetic email examples. Ideal for testing without downloading large datasets.
```bash
python main.py --demo
```

### 2. Single Email Inference
Classify a specific email file using pre-trained models. (Requires running the full pipeline at least once to save models to the `output/models/` directory).
```bash
python main.py --email "path/to/email.txt"
```

### 3. Full Pipeline Mode
Run the complete pipeline on a real dataset.
```bash
python main.py
```
*(Note: Full pipeline mode requires raw dataset folders such as `data/easy_ham/easy_ham` and `data/spam_2/spam_2` to be present).*

## 📊 Outputs

All generated assets are saved to the `output/` directory:
- `output/models/` - Serialized `.pkl` files of the trained models and TF-IDF vectorizer.
- `output/train_clean.csv` & `output/test.csv` - Processed datasets.
- `output/threat_report.txt` - Full text report of the classification run.
- `output/confusion_matrix.png` - Visual evaluation chart of model performance.

## ⚙️ Requirements

- `numpy`
- `pandas`
- `scikit-learn`
- `beautifulsoup4`
- `matplotlib`
- `scipy`
