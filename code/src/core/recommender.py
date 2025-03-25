from core.feature_engineer import create_feature_pipeline
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from typing import List, Dict
import pandas as pd
import logging
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize classifier
        self.risk_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5
        )
        
        # Initialize feature pipeline
        self.feature_pipeline = create_feature_pipeline()
        
        # Initialize other attributes
        self.class_weights = None
        self.sampler = SMOTE(sampling_strategy='auto', k_neighbors=2)
        self.calibrator = None
        self.classes_ = None
        self.label_encoder = LabelEncoder()
        self.use_simple_model = False
    
    def fit(self, X: pd.DataFrame, y: List[str]):
        """Enhanced training with better imbalance handling"""
        try:
            # Encode labels numerically
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            
            # Process features
            X_processed = self.feature_pipeline.fit_transform(X)
            interests_emb = self.embedder.encode(X['Interests'].fillna(''))
            prefs_emb = self.embedder.encode(X['Preferences'].fillna(''))
            features = np.hstack([X_processed, interests_emb, prefs_emb])

            # Check if we have enough samples for proper training
            class_counts = np.bincount(y_encoded)
            if len(class_counts) < 2 or min(class_counts) < 3:
                logger.warning("Insufficient data for proper training. Using simple model.")
                self.use_simple_model = True
                self._train_simple_model(features, y_encoded)
                return

            # Use balanced random forest for better imbalance handling
            self.risk_model = BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                sampling_strategy='not majority',
                random_state=42
            )
            
            self.risk_model.fit(features, y_encoded)
            logger.info("Model trained successfully with balanced random forest")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self.use_simple_model = True
            self._train_simple_model(features, y_encoded)

    def _train_simple_model(self, features, y_encoded):
        """Fallback training for when we have insufficient data"""
        from sklearn.naive_bayes import GaussianNB
        self.simple_model = GaussianNB()
        self.simple_model.fit(features, y_encoded)
        logger.info("Used simple GaussianNB model as fallback")

    def predict_proba(self, X: pd.DataFrame) -> Dict[str, float]:
        """Generate meaningful predictions with proper confidence scores"""
        try:
            # Process features
            X_processed = self.feature_pipeline.transform(X)
            interests_emb = self.embedder.encode(X['Interests'].fillna(''))
            prefs_emb = self.embedder.encode(X['Preferences'].fillna(''))
            features = np.hstack([X_processed, interests_emb, prefs_emb])

            # Get predictions
            if self.use_simple_model:
                probas = self.simple_model.predict_proba(features)[0]
            else:
                probas = self.risk_model.predict_proba(features)[0]

            # Create results dictionary with original class names
            results = {
                cls: float(proba) 
                for cls, proba in zip(self.classes_, probas)
            }
            
            # Apply minimum confidence threshold
            results = {k: max(v, 0.1) for k, v in results.items()}
            
            # Normalize to sum to 1
            total = sum(results.values())
            return {k: v/total for k, v in results.items()}
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            # Return focused fallback recommendations
            return self._focused_fallback(X)

    def _focused_fallback(self, X: pd.DataFrame) -> Dict[str, float]:
        """Intelligent fallback based on customer features"""
        customer = X.iloc[0]
        if customer['amount_mean'] > 50000:
            return {'Investment': 0.6, 'Bank CD': 0.3, 'general': 0.1}
        elif 'home' in str(customer['Interests']).lower():
            return {'Home Loan Repayment': 0.7, 'general': 0.3}
        else:
            return {'general': 1.0}
    
    def get_risk_profile(self, X: pd.DataFrame) -> str:
        """Calculate financial risk profile for a customer"""
        try:
            # Extract relevant features with fallbacks
            amount_mean = float(X['amount_mean'].iloc[0]) if 'amount_mean' in X.columns else 0.0
            income = float(X['Income per year'].iloc[0]) if 'Income per year' in X.columns else 1.0
            sentiment = float(X['sentiment'].iloc[0]) if 'sentiment' in X.columns else 0.5
            
            # Calculate risk ratio (spending vs income)
            risk_ratio = amount_mean / max(income, 1.0)
            
            # Incorporate sentiment (positive sentiment reduces risk)
            risk_score = (risk_ratio * 0.7) + ((1 - sentiment) * 0.3)
            
            # Determine risk profile
            if risk_score > 0.6:
                return "High"
            elif risk_score > 0.3:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.error(f"Risk profile calculation failed: {str(e)}")
            return "Medium"  # Default fallback