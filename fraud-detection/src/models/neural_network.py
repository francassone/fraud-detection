import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, batch_size=256, epochs=50):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        
    def create_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X, y):
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight_dict,
            verbose=0
        )
        return self
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)
    
    def predict_proba(self, X):
        probs = self.model.predict(X)
        return np.hstack([1-probs, probs])