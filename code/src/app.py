from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from core.data_loader import load_data
from core.recommender import HybridRecommender
from core.bias_detector import BiasAuditor
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hyper-Personalized Recommendation API",
    description="Multimodal recommendation engine for financial products",
    version="1.0"
)

# Initialize components
model = HybridRecommender()
auditor = BiasAuditor()
data = None

@app.on_event("startup")
async def startup_event():
    global data
    try:
        logger.info("Loading data...")
        data = load_data()
        
        # Clean column names
        data.columns = data.columns.str.strip()
        
        # Verify required columns
        required_columns = [
            'Customer_Id', 'Age', 'Gender', 'Location', 'Interests',
            'Preferences', 'Income per year', 'Industry', 'Financial Needs',
            'amount_mean', 'preferred_category'
        ]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col != 'preferred_category']
        X = data[feature_columns]
        y = data['preferred_category']
        
        # Train model
        logger.info("Training model...")
        model.fit(X, y)
        logger.info(f"Model trained successfully. Classes: {model.classes_}")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to initialize service: {str(e)}")

class RecommendationRequest(BaseModel):
    customer_id: str
    include_explanation: bool = True

class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[dict]
    risk_profile: str
    bias_check: dict
    success: bool

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    try:
        customer = data[data['Customer_Id'] == request.customer_id]
        if customer.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Get pure model predictions
        probas = model.predict_proba(customer)
        sorted_recs = sorted(probas.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "customer_id": request.customer_id,
            "recommendations": [{
                "product": k,
                "confidence": float(v),
                "explanation": generate_explanation(customer, k) if request.include_explanation else None
            } for k, v in sorted_recs],
            "risk_profile": model.get_risk_profile(customer),
            "bias_check": auditor.check(customer),
            "success": True
        }
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_explanation(customer_data: pd.DataFrame, product: str) -> str:
    customer = customer_data.iloc[0]
    return f"Recommended based on your overall profile characteristics"

@app.get("/health")
async def health_check():
    return {
        "status": "ready" if data is not None else "initializing",
        "model_ready": hasattr(model, 'classes_'),
        "data_records": len(data) if data is not None else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")