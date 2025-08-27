import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import joblib

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import get_preprocessed_data
from src.models.neural_network import KerasClassifier

def evaluate_model(model, X_train, y_train, kfold):
    """Evaluate a model using cross-validation with multiple metrics"""
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    results = {}
    for metric_name, scorer in scoring.items():
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=kfold, scoring=scorer
        )
        results[metric_name] = cv_scores
    
    # Print results
    print(f"\nModel: {model.__class__.__name__}")
    print("-" * 50)
    for metric_name, scores in results.items():
        print(f"{metric_name.capitalize()}:")
        print(f"Mean: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        print(f"Individual fold scores: {np.round(scores, 2)}")
        print("-" * 25)
    
    return results

def train_models(X_train, y_train):
    """Train and evaluate all models"""
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Neural Network': KerasClassifier(
            input_dim=X_train.shape[1],
            batch_size=256,
            epochs=50
        )
    }
    
    # Initialize KFold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate all models
    all_results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        all_results[name] = evaluate_model(model, X_train, y_train, kfold)
    
    # Create summary DataFrame
    summary = pd.DataFrame()
    for model_name, results in all_results.items():
        means = {metric: scores.mean() for metric, scores in results.items()}
        summary = pd.concat([summary, pd.DataFrame(means, index=[model_name])])
    
    # Get best model
    best_model_name = summary['f1'].idxmax()
    best_model = models[best_model_name]
    
    # Train best model on full training set
    best_model.fit(X_train, y_train)
    
    return best_model, summary, all_results

def main():
    # Create directories if they don't exist
    project_root = Path(__file__).parent.parent.resolve()
    data_path = project_root / 'data' / 'creditcard_2023.csv'
    models_dir = project_root / 'models'
    results_dir = project_root / 'results'
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for data file at: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, pca = get_preprocessed_data(str(data_path))
    
    # Train models
    best_model, summary, all_results = train_models(X_train, y_train)
    
    # Save best model and results
    model_path = models_dir / 'best_model.joblib'
    results_path = results_dir / 'model_comparison.csv'
    
    joblib.dump(best_model, model_path)
    summary.round(3).to_csv(results_path)
    
    print("\nTraining completed!")
    print(f"Best model: {summary['f1'].idxmax()}")
    print(f"Best F1 Score: {summary.loc[summary['f1'].idxmax(), 'f1']:.3f}")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()