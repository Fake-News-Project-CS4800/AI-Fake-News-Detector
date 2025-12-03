# AI Fake News Detector

A simple API for detecting AI-generated text using a pre-trained RoBERTa model from HuggingFace.

## 🚀 Quick Start

### 1. Setup (First Time Only)

```bash
# Navigate to project
cd AI-Fake-News-Detector

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install torch transformers fastapi uvicorn pydantic pyyaml nltk captum numpy

# Fix NLTK SSL issue (macOS)
python -c "import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt'); nltk.download('punkt_tab')"

# Install Postgres + SQLAlchemy support
pip install sqlalchemy psycopg2-binary
```
### 2. PostgreSQL Setup

```bash
# Open a terminal with psql (or pgAdmin) and run:
CREATE DATABASE fake_news_db;
CREATE USER fake_user WITH PASSWORD 'fake_password';
GRANT ALL PRIVILEGES ON DATABASE fake_news_db TO fake_user;

# Ensure configuration database URL (configs/api_config.yaml) correlates:
database:
  url: "postgresql+psycopg2://fake_user:fake_password@localhost:5432/fake_news_db"

# Create tables:
source venv/bin/activate
python create_tables.py
```

### 3. Start the API

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Run the API
python run_api.py
```

API runs at: **http://localhost:8000**

### 4. Test the API

**Option A: Browser (easiest)**
- Open http://localhost:8000/docs
- Click "POST /analyze" → "Try it out"
- Paste text and click "Execute"

**Option B: Command line**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to analyze here...",
    "include_explanation": false
  }'
```

**Option C: Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "text": "Your text here...",
        "include_explanation": False
    }
)

result = response.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📋 API Response

```json
{
  "text_hash": "abc123...",
  "label": "AI",                    // "Human", "AI", or "Inconclusive"
  "confidence": 0.98,                // 0-1 confidence score
  "probabilities": {
    "Human": 0.02,
    "AI": 0.98,
    "Inconclusive": 0.0
  },
  "reasons": [
    "Text patterns suggest AI-generated content"
  ],
  "model_version": "1.0.0",
  "processing_time_ms": 25.5
}
```

---

## ⚙️ Configuration

Edit `configs/api_config.yaml`:

```yaml
model:
  name: "Hello-SimpleAI/chatgpt-detector-roberta"  # Pre-trained model
  device: "cpu"                                     # or "cuda" for GPU
  confidence_threshold: 0.7                         # Threshold for "Inconclusive"
```

**Confidence Threshold:**
- `< 0.7` → Classified as "Inconclusive"
- `≥ 0.7` → Classified as "Human" or "AI"

---

## 🔬 Ensemble Mode (Optional)

For improved accuracy, enable ensemble mode using both RoBERTa and Gemini AI:

### 1. Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### 2. Set API Key
Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Restart API
```bash
# Kill current process (if running)
pkill -f "run_api.py"

# Start API (will automatically load .env)
source venv/bin/activate
python run_api.py
```

### 4. Verify Ensemble Mode
Check startup logs - you should see:
```
Gemini detector initialized - ensemble mode enabled
```

**Benefits of Ensemble Mode:**
- ✅ Cross-validation between two different AI models
- ✅ Higher confidence when models agree
- ✅ Flags disagreements for manual review
- ✅ More robust against false positives/negatives
- ✅ See individual predictions from each model

**Without Gemini API key:** System works normally with RoBERTa only (single model).

---

## 🛑 Stop the API

```bash
# Find and kill process
pkill -f "run_api.py"
```

---

## 📁 Project Structure

```
AI-Fake-News-Detector/
├── run_api.py              # ⭐ Use this to start the API
├── configs/
│   ├── api_config.yaml     # API configuration
│   └── model_config.yaml   # Model configuration
├── src/
│   ├── api/app.py          # FastAPI application
│   ├── data/               # Text preprocessing
│   ├── models/             # Model wrappers
│   ├── explainability/     # Explanation generation
│   └── blockchain/         # (TODO: add later)
└── requirements.txt        # Dependencies
```

---

## 🔧 Troubleshooting

### "Connection refused"
API not running. Start it:
```bash
source venv/bin/activate
python run_api.py
```

### "Port 8000 already in use"
Change port in `run_api.py`:
```python
uvicorn.run("src.api.app:app", host="0.0.0.0", port=8001, reload=True)
```

### NLTK errors
```bash
source venv/bin/activate
python -c "import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Model download fails
Check internet connection. Model downloads automatically from HuggingFace on first run.

---

## 📊 Model Information

- **Model:** `Hello-SimpleAI/chatgpt-detector-roberta`
- **Type:** RoBERTa-based binary classifier
- **Labels:** 0=Human, 1=AI (ChatGPT)
- **Free to use:** Yes, from HuggingFace
- **No training required:** Pre-trained and ready

---

## 🎯 What's Working

✅ Real-time text classification
✅ Human vs AI detection
✅ Confidence-based "Inconclusive" category
✅ Fast inference (~20-100ms)
✅ REST API with FastAPI
✅ Interactive API docs
✅ Ensemble mode with RoBERTa + Gemini (optional)
✅ Next.js frontend with real-time analysis
✅ Visual ensemble comparison UI

⚠️ Blockchain integration commented out (add later)

---

## 📝 Next Steps

1. Test with your own text samples
2. Adjust confidence threshold if needed
3. Integrate into your application
4. Add blockchain verification (when ready)

---

## 🤝 Support

- API docs: http://localhost:8000/docs
- Check logs: Watch terminal where `run_api.py` is running
- Test files: `test_ai.json`, `test_human.json` for quick tests
