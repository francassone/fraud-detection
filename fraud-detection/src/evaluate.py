import sys
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import pandas as pd

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import get_preprocessed_data

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print("\nTest Set Performance:")
    print("-" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate and print ROC-AUC Score
    roc_score = float(roc_auc_score(y_test, y_pred_proba))  # Convert to float explicitly
    print(f"\nROC-AUC Score: {roc_score:.3f}")  # Use f-string formatting instead of round()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Path(project_root, 'results', 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        y_pred_proba,
        name='Best Model'
    )
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(Path(project_root, 'results', 'roc_curve.png'))
    plt.close()

def main():
    # Load data
    data_path = Path(project_root, 'data', 'creditcard_2023.csv')
    X_train, X_test, y_train, y_test, scaler, pca = get_preprocessed_data(str(data_path))
    
    # Create results directory if it doesn't exist
    Path(project_root, 'results').mkdir(parents=True, exist_ok=True)
    
    # Load best model
    model_path = Path(project_root, 'models', 'best_model.joblib')
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run train.py first.")
    
    best_model = joblib.load(model_path)
    
    # Evaluate
    evaluate_model(best_model, X_test, y_test)
    
    print("\nEvaluation completed! Check results folder for plots.")

if __name__ == "__main__":
    main()