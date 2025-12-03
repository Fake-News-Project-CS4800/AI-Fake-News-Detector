"""FastAPI application for fake news detection service."""
import os
import time
from typing import Dict, Optional

import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load environment variables from .env file
load_dotenv()

from ..data.preprocessor import compute_hash, normalize_text
from ..explainability.explainer import ModelExplainer
from .gemini_client import GeminiDetector
from .ensemble import EnsembleDetector
# from ..blockchain.proof_packet import ProofPacket  # Blockchain integration - TODO: add later
from sqlalchemy.orm import Session
from ..db.database import SessionLocal
from ..db.models import AnalysisResult

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
#gives the API a connection to the database 

# Load API config
with open('./configs/api_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

api_config = config['api']
model_config = config['model']
# blockchain_config = config['blockchain']  # Blockchain integration - TODO: add later

# Initialize FastAPI app
app = FastAPI(
    title="AI Fake News Detector API",
    description="API for detecting AI-generated fake news using transformer models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AnalyzeRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., min_length=10, description="Text to analyze")
    include_explanation: bool = Field(default=True, description="Include explanation in response")
    # include_proof_packet: bool = Field(default=True, description="Include blockchain proof packet")  # Blockchain - TODO: add later


class AnalyzeResponse(BaseModel):
    """Response model for text analysis."""
    text_hash: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    reasons: list[str]
    model_version: str
    processing_time_ms: float
    # proof_packet: Optional[Dict] = None  # Blockchain - TODO: add later
    explanation: Optional[Dict] = None
    ensemble: Optional[Dict] = None  # Ensemble data if available


# Global variables for model (loaded at startup)
model = None
tokenizer = None
explainer = None
gemini_detector = None
ensemble_detector = None
device = None
MODEL_VERSION = "1.0.0"


@app.on_event("startup")
async def load_model():
    """Load model at startup."""
    global model, tokenizer, explainer, gemini_detector, ensemble_detector, device

    print("Loading model...")

    device = torch.device(model_config['device'])
    model_name = model_config.get('name', 'Hello-SimpleAI/chatgpt-detector-roberta')

    # Load pre-trained model and tokenizer from HuggingFace
    print(f"Loading {model_name} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Initialize explainer (optional - may fail with some models)
    try:
        explainer = ModelExplainer(model, tokenizer, device)
        print("Explainer initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize explainer: {e}")
        print("API will work but explanations will be limited")
        explainer = None

    # Initialize Gemini detector (optional - requires API key)
    try:
        gemini_detector = GeminiDetector()
        ensemble_detector = EnsembleDetector()
        print("Gemini detector initialized - ensemble mode enabled")
    except Exception as e:
        print(f"Warning: Could not initialize Gemini detector: {e}")
        print("API will work with RoBERTa only (no ensemble)")
        gemini_detector = None
        ensemble_detector = None

    print(f"Model loaded successfully on {device}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AI Fake News Detector",
        "version": MODEL_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# accepts the user’s text and gives the function a DB connection 

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeRequest,
    db: Session = Depends(get_db)
):

    """Analyze text for AI-generated content.

    Args:
        request: Analysis request

    Returns:
        Analysis response with prediction, explanation, and proof packet
    """
    start_time = time.time()

    try:
        # Normalize and hash text
        normalized_text = normalize_text(request.text)
        text_hash = compute_hash(request.text)

        # Tokenize
        encoding = tokenizer(
            normalized_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            # Model outputs 2 classes: [Human, AI]
            probs = torch.softmax(logits, dim=-1)
            predicted_class_2way = torch.argmax(probs, dim=-1).item()
            confidence_2way = probs[0, predicted_class_2way].item()

        # Convert 2-class to 3-class system using confidence threshold
        confidence_threshold = model_config.get('confidence_threshold', 0.7)

        if confidence_2way < confidence_threshold:
            # Low confidence -> Inconclusive
            predicted_class = 2  # Inconclusive
            label = 'Inconclusive'
            confidence = 1.0 - confidence_2way  # Confidence in being inconclusive
        else:
            # High confidence -> Human or AI
            predicted_class = predicted_class_2way
            label = 'Human' if predicted_class_2way == 0 else 'AI'
            confidence = confidence_2way

        # Get probabilities for all 3 classes
        human_prob = float(probs[0, 0])
        ai_prob = float(probs[0, 1])

        # Calculate inconclusive probability based on uncertainty
        max_prob = max(human_prob, ai_prob)
        inconclusive_prob = 1.0 - max_prob if max_prob < confidence_threshold else 0.0

        # Normalize probabilities to sum to 1
        total = human_prob + ai_prob + inconclusive_prob
        probabilities = {
            'Human': human_prob / total,
            'AI': ai_prob / total,
            'Inconclusive': inconclusive_prob / total
        }

        # Store RoBERTa prediction for ensemble
        roberta_prediction = {
            'label': label,
            'confidence': confidence,
            'probabilities': probabilities
        }

        # Get Gemini prediction and combine with ensemble if available
        ensemble_result = None
        if gemini_detector is not None and ensemble_detector is not None:
            try:
                print("Getting Gemini prediction...")
                gemini_prediction = gemini_detector.analyze_text(request.text)

                print(f"Gemini: {gemini_prediction['label']} ({gemini_prediction['confidence']:.2%})")
                print(f"RoBERTa: {label} ({confidence:.2%})")

                # Combine predictions
                ensemble_result = ensemble_detector.combine_predictions(
                    roberta_prediction,
                    gemini_prediction
                )

                # Update final prediction with ensemble result
                label = ensemble_result['label']
                confidence = ensemble_result['confidence']

                # Update probabilities based on ensemble (if not inconclusive)
                if label != 'Inconclusive':
                    probabilities = ensemble_result['roberta']['probabilities']

                print(f"Ensemble: {label} ({confidence:.2%}) - {ensemble_result['agreement_level']}")

            except Exception as e:
                print(f"Warning: Ensemble prediction failed: {e}")
                print("Falling back to RoBERTa-only prediction")
                ensemble_result = None

        # Get explanation if requested
        # TEMPORARILY DISABLE EXPLANATIONS (otherwise Captum errors out)
        explanation = None
        reasons = [f"Classification: {label} (confidence: {confidence:.2%})"]


        if explainer is not None:
            try:
                if request.include_explanation:
                    full_explanation = explainer.explain_prediction(request.text)
                    base_reasons = full_explanation['reasons']
                    explanation = {
                        'top_tokens': full_explanation['top_tokens'][:5],  # Limit for API response
                        'full_analysis': full_explanation
                    }
                else:
                    # Still get basic reasons
                    base_reasons = explainer.get_heuristic_reasons(request.text, predicted_class)

                # Add ensemble reasons if available
                if ensemble_result is not None:
                    reasons = ensemble_detector.get_ensemble_reasons(ensemble_result, base_reasons)
                else:
                    reasons = base_reasons

            except Exception as e:
                print(f"Warning: Explanation failed: {e}")
                reasons = [f"Classification: {label} (confidence: {confidence:.2%})"]
        else:
            if ensemble_result is not None:
                reasons = ensemble_detector.get_ensemble_reasons(ensemble_result, [])
            else:
                reasons = [f"Classification: {label} (confidence: {confidence:.2%})"]

        # TODO: Blockchain integration - Generate proof packet here later
        # proof_packet = None
        # if request.include_proof_packet:
        #     proof_packet = ProofPacket.create_packet(
        #         text_hash=text_hash,
        #         label=label,
        #         confidence=confidence,
        #         model_version=MODEL_VERSION,
        #         reasons=reasons
        #     )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Save result into the database
        db_obj = AnalysisResult(
            text_hash=text_hash,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            reasons=reasons,
            model_version=MODEL_VERSION,  
            raw_text=request.text
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)

        return AnalyzeResponse(
            text_hash=text_hash,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            reasons=reasons,
            model_version=MODEL_VERSION,
            processing_time_ms=processing_time_ms,
            # proof_packet=proof_packet,  # Blockchain - TODO: add later
            explanation=explanation,
            ensemble=ensemble_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch-analyze")
async def batch_analyze(texts: list[str]):
    """Batch analyze multiple texts.

    Args:
        texts: List of texts to analyze

    Returns:
        List of analysis results
    """
    results = []

    for text in texts:
        try:
            request = AnalyzeRequest(text=text, include_explanation=False)
            result = await analyze_text(request)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "text_preview": text[:50] + "..."
            })

    return {"results": results, "total": len(texts)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=api_config['host'],
        port=api_config['port'],
        reload=api_config['reload'],
        workers=api_config.get('workers', 1)
    )
