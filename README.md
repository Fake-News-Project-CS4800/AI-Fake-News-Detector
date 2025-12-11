# AI Fake News Detector

A full-stack application for detecting AI-generated text using machine learning.

> **📍 Deployment Note**: This application runs **locally only**. The ML model (PyTorch + RoBERTa) requires ~2GB RAM, which exceeds free hosting limits. The frontend can be deployed to Vercel for demonstration purposes, but the backend must run on your local machine. See [Deployment](#-deployment) for details.

## 🌟 Overview

**Backend**: FastAPI + RoBERTa transformer model + Gemini AI ensemble
**Frontend**: Next.js 14 with React 19, interactive analysis tools
**Features**: Real-time detection, style analysis, confidence calibration, robustness testing

### Tech Stack
- 🧠 **ML Models**: RoBERTa (HuggingFace) + Google Gemini 2.5 Flash
- ⚡ **Backend**: Python, FastAPI, PyTorch, PostgreSQL
- 🎨 **Frontend**: Next.js 14, React 19, TypeScript, Tailwind CSS v4
- 📊 **Analysis**: 7 NLP metrics, adversarial testing, threshold tuning

---

## 🚀 Quick Start

### 1. Backend Setup

```bash
# Navigate to project
cd AI-Fake-News-Detector

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Fix NLTK SSL issue (macOS)
python -c "import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt'); nltk.download('punkt_tab')"
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

### 3. Setup Frontend (Optional but Recommended)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Go back to root
cd ..
```

### 4. Start the API

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Run the API
python run_api.py
```

API runs at: **http://localhost:8000**

### 5. Start Frontend (Optional - for UI)

**In a new terminal:**
```bash
cd frontend
npm run dev
```

Frontend runs at: **http://localhost:3000** 🎨

### 6. Test the Application

**Option A: Web Interface (Recommended)**
- Open http://localhost:3000
- Paste text into the textbox
- Click send to analyze
- Explore style analysis, calibration, and robustness features!

**Option B: API Docs (Interactive Swagger UI)**
- Open http://localhost:8000/docs
- Click "POST /analyze" → "Try it out"
- Paste text and click "Execute"

**Option C: Command line**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text to analyze here...",
    "include_explanation": false
  }'
```

**Option D: Python**
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

## 🎨 Frontend Setup (Next.js)

The project includes a modern Next.js frontend with real-time analysis and interactive visualizations.

### 1. Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install
```

### 2. Configure API URL

Create `.env.local` in the `frontend/` directory:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Start Frontend Development Server

```bash
# From frontend/ directory
npm run dev
```

Frontend runs at: **http://localhost:3000**

### 4. Build for Production

```bash
# From frontend/ directory
npm run build
npm start
```

---

## 🖥️ Frontend Features

### **Main Interface**
- **Real-time text analysis**: Paste text and get instant AI detection results
- **Chat-style interface**: Newest results appear at top, input box stays fixed
- **Sticky input area**: Never lose the textbox when scrolling through results

### **Result Display**
- **Classification badge**: Human 🟢 / AI 🔴 / Inconclusive 🟡
- **Confidence score**: Percentage confidence for each prediction
- **"Most likely" indicator**: Shows lean when result is Inconclusive
- **Probability bars**: Visual breakdown of Human/AI/Inconclusive probabilities
- **Processing time**: Shows analysis speed in milliseconds

### **Ensemble Analysis** (when Gemini API is enabled)
- **Dual model comparison**: Side-by-side RoBERTa vs Gemini predictions
- **Agreement indicator**: Shows if models agree (Strong/Weak/Disagreement)
- **Individual confidences**: See each model's confidence separately
- **Gemini reasoning**: Text explanation from Gemini's analysis

### **Advanced Features**

#### 📊 **Writing Style Analysis**
Click "View Writing Style Analysis" to see:
- **Lexical Diversity**: Type-Token Ratio (vocabulary richness)
- **Perplexity Proxy**: Entropy and text predictability
- **Sentence Complexity**: Average length, variation, complex sentences
- **Readability Scores**: Flesch Reading Ease, grade level
- **Vocabulary Richness**: Word length distribution
- **Punctuation Patterns**: Comma, period, exclamation usage
- **Text Statistics**: Character, word, sentence counts

Each metric includes:
- Visual progress bar
- Score interpretation
- "What it means" educational box with formulas

#### ⚙️ **Confidence Calibration Explorer**
Interactive threshold tuning:
- **Adjustable slider**: Change confidence threshold (0-100%)
- **Real-time updates**: See how classification changes at different thresholds
- **Precision/Recall tradeoff**: Understand false positive vs false negative rates
- **Threshold comparison table**: View results at common thresholds (50%, 60%, 70%, 80%, 90%)
- **Educational explanations**: Learn about threshold selection for different use cases

#### 🛡️ **Model Robustness Testing**
Tests model stability against realistic text variations:
- **Natural typos**: Common misspellings, doubled/missing letters
- **Punctuation variations**: Extra commas, ellipses, missing punctuation
- **Whitespace errors**: Extra spaces, missing spaces after punctuation
- **Filler words**: "actually", "basically", "you know", "like"
- **Contractions**: can't ↔ cannot, it's ↔ it is
- **Capitalization**: All lowercase, missing initial capitals
- **Article changes**: Adding/removing "the", "a", "an"

Results include:
- **Robustness score**: 0-100% stability rating
- **Label flip count**: How many variations changed the prediction
- **Detailed test results**: Each variation with before/after comparison
- **Educational content**: Explains why robustness matters

### **UI/UX Details**
- **Gradient backgrounds**: Dark gray gradients for modern aesthetic
- **Responsive design**: Works on desktop, tablet, mobile
- **Loading states**: Animated spinners during analysis
- **Error handling**: Clear error messages with retry options
- **Modal interfaces**: Click-outside-to-close for all dialogs
- **Smooth animations**: Transitions and hover effects

---

## 📁 Project Structure

```
AI-Fake-News-Detector/
├── run_api.py              # ⭐ Use this to start the API
├── configs/
│   ├── api_config.yaml     # API configuration
│   └── model_config.yaml   # Model configuration
├── src/
│   ├── api/
│   │   ├── app.py          # FastAPI application
│   │   ├── gemini_client.py # Gemini API integration
│   │   └── ensemble.py     # Ensemble decision logic
│   ├── data/               # Text preprocessing
│   ├── models/             # Model wrappers
│   ├── explainability/     # Explanation generation
│   ├── analysis/
│   │   ├── style_analyzer.py      # NLP metrics analysis
│   │   └── adversarial_tester.py  # Robustness testing
│   ├── db/                 # Database models and connection
│   └── blockchain/         # (TODO: add later)
├── frontend/               # Next.js web interface
│   ├── app/                # Next.js 13+ App Router
│   ├── components/
│   │   ├── ChatInterface.tsx           # Main chat UI
│   │   ├── ResultCard.tsx              # Result display
│   │   ├── StyleAnalysisModal.tsx      # Style metrics modal
│   │   ├── ConfidenceCalibrationModal.tsx  # Threshold explorer
│   │   └── AdversarialTestingModal.tsx # Robustness testing UI
│   ├── lib/
│   │   ├── api.ts          # API client functions
│   │   └── types.ts        # TypeScript type definitions
│   ├── package.json        # Frontend dependencies
│   └── tailwind.config.ts  # Tailwind CSS config
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variables template
```

---

## 🔧 Troubleshooting

### Backend Issues

#### "Connection refused"
API not running. Start it:
```bash
source venv/bin/activate
python run_api.py
```

#### "Port 8000 already in use"
Change port in `run_api.py`:
```python
uvicorn.run("src.api.app:app", host="0.0.0.0", port=8001, reload=True)
```

#### NLTK errors
```bash
source venv/bin/activate
python -c "import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('punkt'); nltk.download('punkt_tab')"
```

#### Model download fails
Check internet connection. Model downloads automatically from HuggingFace on first run.

### Frontend Issues

#### "Failed to connect to the API"
1. Make sure backend is running on port 8000
2. Check `.env.local` has correct API URL:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
3. Verify CORS is enabled in backend (already configured)

#### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

#### "Port 3000 already in use"
Kill the process or change port:
```bash
# Kill existing process
lsof -ti:3000 | xargs kill

# Or run on different port
PORT=3001 npm run dev
```

#### Style/UI issues
Clear Next.js cache:
```bash
cd frontend
rm -rf .next
npm run dev
```

---

## 📊 Model Information

- **Model:** `Hello-SimpleAI/chatgpt-detector-roberta`
- **Type:** RoBERTa-based binary classifier
- **Labels:** 0=Human, 1=AI (ChatGPT)
- **Free to use:** Yes, from HuggingFace
- **No training required:** Pre-trained and ready

---

## 🎯 What's Working

### Backend API
✅ Real-time text classification (Human/AI/Inconclusive)
✅ RoBERTa-based transformer model
✅ Confidence-based thresholding
✅ Fast inference (~20-100ms)
✅ REST API with FastAPI
✅ Interactive API docs at `/docs`
✅ Ensemble mode with RoBERTa + Gemini (optional)
✅ PostgreSQL database integration
✅ Writing style analysis (7 NLP metrics)
✅ Model robustness testing (20+ realistic variations)

### Frontend (Next.js)
✅ Modern chat-style interface
✅ Real-time text analysis
✅ Sticky input textbox (stays at top)
✅ Newest results appear first
✅ Visual probability bars
✅ Ensemble comparison UI (when enabled)
✅ "Most likely" indicator for Inconclusive results
✅ Interactive writing style analysis modal
✅ Confidence calibration explorer with slider
✅ Model robustness testing with detailed results
✅ Educational tooltips and explanations
✅ Responsive design (mobile/tablet/desktop)
✅ Dark gradient theme
✅ Loading states and error handling

### Advanced Features
✅ 7-metric style analyzer (TTR, entropy, readability, etc.)
✅ Precision/recall tradeoff visualization
✅ Realistic adversarial testing (typos, punctuation, fillers)
✅ Interactive threshold adjustment
✅ Robustness scoring (0-100%)
✅ Click-outside-to-close modals

⚠️ Blockchain integration commented out (add later)

---

## 🚀 Running the Full Stack

To run both backend and frontend together:

### Terminal 1 - Backend
```bash
source venv/bin/activate
python run_api.py
# API runs at http://localhost:8000
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
# Frontend runs at http://localhost:3000
```

Now visit **http://localhost:3000** to use the full application!

---

## 🌐 Deployment

### Frontend Deployment (Vercel)

The frontend can be deployed to Vercel for demonstration:

```bash
cd frontend
vercel
```

**Important**: Set environment variable in Vercel:
- `NEXT_PUBLIC_API_URL` = `http://localhost:8000`

**Note**: The deployed frontend will only work when:
1. You're accessing it from the same machine where the backend runs
2. The backend is running locally

This is suitable for portfolio/demo purposes but won't work for external users.

### Backend Deployment

**The backend cannot be deployed to free hosting services** due to memory requirements:

- **Memory needed**: ~2GB RAM (for PyTorch + RoBERTa model)
- **Free tier limits**: 512MB (Render, Railway, Heroku)
- **Paid options**:
  - Render ($7/mo for 2GB)
  - Railway ($10/mo)
  - DigitalOcean ($12/mo)
  - AWS/GCP (pay per use)

**For class projects/demos**: Running locally is perfectly acceptable and standard practice for ML applications.

**Alternative**: Use only Gemini API (no local model) for a lighter backend that can be deployed free, but you'll lose the RoBERTa ensemble and some features.

---

## 📝 Next Steps

1. ✅ Test with your own text samples via the frontend
2. ✅ Explore the style analysis, calibration, and robustness features
3. ✅ Adjust confidence threshold in `configs/api_config.yaml` if needed
4. ✅ Enable Gemini ensemble mode for improved accuracy (optional)
5. 📋 Add custom features or integrate into your application
6. 📋 Add blockchain verification (when ready)

---

## 🤝 Support

- **Frontend**: http://localhost:3000
- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Logs**: Watch terminals where backend/frontend are running
- **Test files**: `test_ai.json`, `test_human.json` for quick API tests

---

## 📚 Key Technologies

- **Backend**: Python 3.x, FastAPI, PyTorch, Transformers, Google Gemini API
- **Frontend**: Next.js 14, React 19, TypeScript, Tailwind CSS v4
- **Database**: PostgreSQL, SQLAlchemy
- **ML Models**: RoBERTa (HuggingFace), Gemini 2.5 Flash (Google)
- **Analysis**: NLTK, TextStat, NumPy
- **Hosting**: Local development (backend), Vercel (frontend demo)
