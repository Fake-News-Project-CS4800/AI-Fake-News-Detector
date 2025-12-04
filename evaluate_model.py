"""Comprehensive evaluation of the fine-tuned model."""
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation.metrics import compute_metrics
import os

def evaluate_fine_tuned_model():
    """Evaluate the fine-tuned model comprehensively."""
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    # Check if fine-tuned model exists
    model_path = "./models/checkpoints/best_model"
    if not os.path.exists(model_path):
        print("No fine-tuned model found!")
        print("Available models in models/checkpoints/:")
        if os.path.exists("./models/checkpoints"):
            for item in os.listdir("./models/checkpoints"):
                print(f"  - {item}")
        return
    
    # Load test data
    test_path = './data/splits/test.csv'
    if not os.path.exists(test_path):
        print("Test data not found! Please run prepare_dataset.py first.")
        return
    
    test_df = pd.read_csv(test_path)
    print(f"Test dataset: {len(test_df)} samples")
    print(f"Label distribution:\n{test_df['label'].value_counts()}")
    
    # Load fine-tuned model
    print(f"\nLoading fine-tuned model from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Use MPS if available for faster inference
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"Model loaded on {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run predictions
    print("\nRunning predictions...")
    predictions = []
    probabilities = []
    confidence_scores = []
    
    # Process in smaller batches for stability
    batch_size = 32
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df['text'].iloc[i:i+batch_size].tolist()
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Get confidence (max probability)
            max_probs = torch.max(probs, dim=-1)[0]
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            confidence_scores.extend(max_probs.cpu().numpy())
        
        if (i + batch_size) % 100 == 0:
            print(f"  Processed {min(i + batch_size, len(test_df))}/{len(test_df)} samples")
    
    # Convert to arrays
    y_true = test_df['label'].values
    y_pred = np.array(predictions)
    y_proba = np.array(probabilities)
    confidences = np.array(confidence_scores)
    
    # Calculate metrics
    print("\nEVALUATION RESULTS")
    print("=" * 40)
    
    # 1. Overall Performance
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"Overall Performance:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  F1 Score (Macro):   {f1_macro:.4f}")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"  Precision (Macro):  {precision_macro:.4f}")
    print(f"  Recall (Macro):     {recall_macro:.4f}")
    
    # 2. Per-class Performance
    print(f"\nPer-Class Performance:")
    class_names = ['Human', 'AI']
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Human    AI")
    print(f"Actual Human     {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"Actual AI        {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # 4. Error Analysis
    print(f"\nError Analysis:")
    false_positives = np.sum((y_true == 0) & (y_pred == 1))  # Human → AI
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))  # AI → Human
    print(f"  False Positives (Human→AI): {false_positives}")
    print(f"  False Negatives (AI→Human): {false_negatives}")
    print(f"  Total Errors: {false_positives + false_negatives}")
    
    # 5. Confidence Analysis
    human_confidences = confidences[y_pred == 0]
    ai_confidences = confidences[y_pred == 1]
    
    print(f"\nConfidence Analysis:")
    print(f"  Human predictions:")
    print(f"    Average confidence: {np.mean(human_confidences):.4f} ± {np.std(human_confidences):.4f}")
    print(f"    Min confidence: {np.min(human_confidences):.4f}")
    print(f"    Max confidence: {np.max(human_confidences):.4f}")
    
    print(f"  AI predictions:")
    print(f"    Average confidence: {np.mean(ai_confidences):.4f} ± {np.std(ai_confidences):.4f}")
    print(f"    Min confidence: {np.min(ai_confidences):.4f}")
    print(f"    Max confidence: {np.max(ai_confidences):.4f}")
    
    # 6. Low Confidence Predictions
    low_confidence_threshold = 0.6
    low_confidence_mask = confidences < low_confidence_threshold
    low_confidence_count = np.sum(low_confidence_mask)
    
    print(f"\nLow Confidence Analysis (< {low_confidence_threshold}):")
    print(f"  Count: {low_confidence_count}/{len(confidences)} ({100*low_confidence_count/len(confidences):.1f}%)")
    
    if low_confidence_count > 0:
        low_conf_accuracy = accuracy_score(y_true[low_confidence_mask], y_pred[low_confidence_mask])
        print(f"  Accuracy on low-confidence predictions: {low_conf_accuracy:.4f}")
    
    # 7. Sample Error Examples
    print(f"\nERROR EXAMPLES:")
    print("=" * 40)
    
    # Find errors
    error_mask = y_true != y_pred
    error_indices = np.where(error_mask)[0]
    
    # Show up to 5 error examples
    for i, idx in enumerate(error_indices[:5]):
        true_label = "Human" if y_true[idx] == 0 else "AI"
        pred_label = "Human" if y_pred[idx] == 0 else "AI"
        confidence = confidences[idx]
        text_preview = test_df.iloc[idx]['text'][:150] + "..."
        
        print(f"\nExample {i+1}:")
        print(f"  True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.3f}")
        print(f"  Text: {text_preview}")
    
    # 8. Model Comparison
    print(f"\nMODEL COMPARISON:")
    print("=" * 40)
    
    # Compare with random baseline
    random_accuracy = 0.5  # For binary classification
    print(f"Random Baseline:     {random_accuracy:.4f}")
    print(f"Your Model:          {accuracy:.4f}")
    print(f"Improvement:         +{accuracy - random_accuracy:.4f} ({100*(accuracy - random_accuracy):.1f}%)")
    
    # Performance categories
    if accuracy >= 0.95:
        performance_level = "Excellent"
    elif accuracy >= 0.90:
        performance_level = "Very Good"
    elif accuracy >= 0.85:
        performance_level = "Good"
    elif accuracy >= 0.80:
        performance_level = "Fair"
    elif accuracy >= 0.70:
        performance_level = "Poor"
    else:
        performance_level = "Very Poor"
    
    print(f"Performance Level:   {performance_level}")
    
    # 9. Save results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'low_confidence_count': low_confidence_count,
        'avg_confidence_human': np.mean(human_confidences),
        'avg_confidence_ai': np.mean(ai_confidences)
    }
    
    # Save to file
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to evaluation_results.json")
    print(f"\nEvaluation complete!")

if __name__ == "__main__":
    evaluate_fine_tuned_model()