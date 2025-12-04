"""Train/fine-tune the AI detection model."""
import sys
import os
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import print_evaluation_report

def main():
    """Main training function."""
    print("Starting model fine-tuning...")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer("./configs/model_config.yaml")
    
    print(f"Model: {trainer.model_config['name']}")
    print(f"Device: {trainer.device}")
    print(f"Batch size: {trainer.train_config['batch_size']}")
    print(f"Learning rate: {trainer.train_config['learning_rate']}")
    print(f"Epochs: {trainer.train_config['num_epochs']}")
    
    # Check if data exists
    train_path = trainer.data_config['train_path']
    if not os.path.exists(train_path):
        print(f"❌ Training data not found at {train_path}")
        print("Please run: python prepare_dataset.py")
        return
    
    print("\n🚀 Starting training...")
    trainer.train()
    
    print("\n✅ Training completed!")
    print(f"Best model saved to: {trainer.train_config['output_dir']}/best_model")
    
    # Evaluate on test set if available
    test_path = trainer.data_config['test_path']
    if os.path.exists(test_path):
        print("\n📊 Evaluating on test set...")
        _, val_loader = trainer.load_data()  # Use validation for now
        metrics = trainer.evaluate(val_loader)
        
        print(f"Final Test Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_macro']:.4f}")
        print(f"Precision: {metrics['precision_macro']:.4f}")
        print(f"Recall: {metrics['recall_macro']:.4f}")

if __name__ == "__main__":
    main()