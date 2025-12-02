# AI Fake News Detector - Frontend

Clean, modern UI for detecting AI-generated text.

## 🚀 Quick Start

**1. Make sure the backend is running first:**

```bash
# In the root directory
source venv/bin/activate
python run_api.py
```

**2. Start the frontend:**

```bash
# In the frontend directory
npm run dev
```

**3. Open http://localhost:3000** in your browser

## ⚙️ Configuration

The backend API URL is configured in `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📁 Project Structure

```
frontend/
├── app/
│   ├── page.tsx          # Main page with ChatInterface
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
├── components/
│   ├── ChatInterface.tsx # Chat-style UI (user input + results)
│   └── ResultCard.tsx    # Display AI/Human classification
└── lib/
    ├── api.ts            # API client (calls FastAPI)
    └── types.ts          # TypeScript interfaces
```

## 🎨 Features

- ✅ Clean, intuitive interface
- ✅ Real-time text analysis
- ✅ Visual confidence meters
- ✅ Detailed explanations
- ✅ Responsive design
- ✅ TypeScript + Tailwind CSS

## 🔧 Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## 💡 How to Use

1. Paste or type text into the input area
2. Click "Analyze"
3. View the results:
   - Label (Human/AI/Inconclusive)
   - Confidence score
   - Probability breakdown
   - Explanation/reasons
