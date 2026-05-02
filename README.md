# AI-Powered Phishing Detector

A complete, end-to-end Machine Learning pipeline for detecting and classifying phishing emails. This cybersecurity project uses both hand-crafted threat indicators and TF-IDF statistical features to train models (Logistic Regression and Random Forest) to identify malicious emails.

Now features a **Chrome Extension** for real-time protection in Gmail.

## 🚀 Pipeline Components

The system is highly modular and broken down into 5 sequential components:

1. **Data Ingestion (`component1_data_ingestion.py`)**  
   Loads raw emails from disk, assigns labels, and splits data.
2. **Pre-Processing (`component2_preprocessing.py`)**  
   Sanitizes raw email text by stripping HTML, tokenizing URLs, and removing stop words.
3. **Threat Signal Extraction (`component3_feature_extraction.py`)**  
   Extracts 7 hand-crafted attack indicators and TF-IDF statistical features.
4. **ML Classification Engine (`component4_classification.py`)**  
   Trains and evaluates Logistic Regression and Random Forest models.
5. **Threat Reporting (`component5_reporting.py`)**  
   Generates human-readable reports and overall security metrics.

---

## 🌐 Chrome Extension Integration (New)

You can now use this project as a real-time scanner for your actual Gmail inbox.

### 1. Start the API Server
Ensure you have trained your models (run `python main.py` at least once). Then start the backend:
```bash
python app.py
```
The API will run at `http://localhost:8000`.

### 2. Install the Extension
1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the `extension/` folder in this repository.
4. Refresh your Gmail.

### 3. Features
- **Real-time Scanning:** Automatically scans emails as you open them.
- **Premium UI Banner:** Displays a sleek, luxury-themed security verdict (PHISH ⚠ or SECURE ✅) directly in the Gmail interface.
- **Live Logs:** View detailed threat scores and triggered indicators in your terminal.

---

## 🛠️ CLI Usage

The orchestrator script (`main.py`) supports three execution modes:

### 1. Demo Mode
```bash
python main.py --demo
```

### 2. Single Email Inference
```bash
python main.py --email "path/to/email.txt"
```

### 3. Full Pipeline Mode
```bash
python main.py
```

## 📊 Outputs

- `output/models/` - Serialized `.pkl` files (Model + Vectorizer).
- `output/threat_report.txt` - Full text report of the classification run.
- `output/confusion_matrix.png` - Performance visualization.

## ⚙️ Requirements

- `fastapi`, `uvicorn` (for the API)
- `numpy`, `pandas`, `scikit-learn`
- `beautifulsoup4`
- `matplotlib`, `scipy`
