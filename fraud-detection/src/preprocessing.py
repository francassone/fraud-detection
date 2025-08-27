import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

def load_and_preprocess_data(filepath):
    """Load and perform initial preprocessing of the data"""
    # Load data
    df = pd.read_csv(filepath)
    
    # Drop ID and create log transform of Amount
    df = df.drop(columns=['id'])
    df['Amount_log'] = np.log1p(df['Amount'])
    
    return df

def split_data(df):
    """Split data into features and target, then into train and test sets"""
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, columns_to_scale=['Amount', 'Amount_log']):
    """Scale specified features using StandardScaler"""
    scaler = StandardScaler()
    
    # Create copies to avoid modifying the original data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit and transform training data
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    # Transform test data
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    return X_train_scaled, X_test_scaled, scaler

def apply_pca(X_train, X_test, n_components=10):
    """Apply PCA transformation to the data"""
    # First PCA to analyze components
    pca_initial = FactorAnalyzer(
        n_factors=30,
        rotation=None,
        method='principal'
    ).fit(X_train)
    
    # Final PCA with selected components
    pca_final = FactorAnalyzer(
        n_factors=n_components,
        rotation='varimax',
        method='principal'
    ).fit(X_train)
    
    # Transform both sets
    X_train_pca = pca_final.transform(X_train)
    X_test_pca = pca_final.transform(X_test)
    
    return X_train_pca, X_test_pca, pca_final

def get_preprocessed_data(data_filepath, n_pca_components=10):
    """Main function to run the entire preprocessing pipeline"""
    # Load and preprocess
    df = load_and_preprocess_data(data_filepath)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Apply PCA
    X_train_final, X_test_final, pca = apply_pca(
        X_train_scaled, 
        X_test_scaled, 
        n_components=n_pca_components
    )
    
    return X_train_final, X_test_final, y_train, y_test, scaler, pca

if __name__ == "__main__":
    # Example usage
    DATA_PATH = "../data/creditcard_2023.csv"
    X_train, X_test, y_train, y_test, scaler, pca = get_preprocessed_data(DATA_PATH)
    print("Preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")