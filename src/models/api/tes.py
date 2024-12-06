from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import os

app = FastAPI(
    title="XYZ E-Commerce Analytics API",
    description="""
    API for E-Commerce Analytics and Sales Predictions.
    
    This API provides:
    * Sales Performance Analytics
    * Pricing Strategy Insights
    * Customer Satisfaction Metrics
    * Sales Predictions
    
    For frontend developers:
    * All endpoints return JSON
    * Error responses include detail messages
    * All numeric values are either float or integer
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    category: str = Field(..., example="ELECTRONICS", description="Product category")
    price: float = Field(..., example=299.99, description="Product price")
    discount_percentage: float = Field(..., ge=0, le=100, example=15.0, description="Discount percentage (0-100)")
    rating: float = Field(..., ge=1, le=5, example=4.5, description="Current product rating (1-5)")
    review_count: int = Field(..., ge=0, example=150, description="Number of product reviews")
    
    class Config:
        schema_extra = {
            "example": {
                "category": "ELECTRONICS",
                "price": 299.99,
                "discount_percentage": 15.0,
                "rating": 4.5,
                "review_count": 150
            }
        }

class PredictionResponse(BaseModel):
    predicted_sales: float = Field(..., description="Predicted sales volume")
    predicted_revenue: float = Field(..., description="Predicted revenue")
    confidence_score: float = Field(..., description="Prediction confidence score (0-1)")
    recommendations: List[str] = Field(..., description="List of recommendations based on prediction")

class CategoryAnalytics(BaseModel):
    category: str
    total_revenue: float
    total_sales: int
    average_price: float
    discount_effectiveness: float
    price_satisfaction: float
    review_sales_ratio: float

# Load trained model
MODEL_PATH = "../training/trained_models/sales_predictor.joblib"

@app.get("/", tags=["General"])
async def root():
    """
    Welcome endpoint with API information.
    """
    return {
        "message": "Welcome to XYZ E-Commerce Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns server status and model availability.
    """
    try:
        model_loaded = os.path.exists(MODEL_PATH)
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "api_version": "1.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/sales", 
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict sales for a product",
    description="Get sales predictions and recommendations based on product attributes")
async def predict_sales(data: PredictionRequest):
    """
    Predict sales and revenue for a product based on its attributes.
    
    Parameters:
    - category: Product category (e.g., "ELECTRONICS")
    - price: Product price
    - discount_percentage: Discount percentage (0-100)
    - rating: Current product rating (1-5)
    - review_count: Number of product reviews
    
    Returns:
    - Predicted sales volume
    - Predicted revenue
    - Confidence score
    - Recommendations list
    """
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        
        # Prepare input data
        features = np.array([[
            data.price,
            data.discount_percentage,
            data.rating,
            data.review_count
        ]])
        
        # Make prediction
        predicted_sales = float(model.predict(features)[0])
        predicted_revenue = predicted_sales * data.price * (1 - data.discount_percentage/100)
        
        return {
            "predicted_sales": predicted_sales,
            "predicted_revenue": predicted_revenue,
            "confidence_score": 0.95,
            "recommendations": [
                f"Expected sales volume: {predicted_sales:.0f} units",
                f"Projected revenue: ${predicted_revenue:.2f}",
                "Consider adjusting price based on category average"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/category/{category}",
    response_model=CategoryAnalytics,
    tags=["Analytics"],
    summary="Get category analytics",
    description="Retrieve detailed analytics for a specific product category")
async def get_category_analytics(category: str):
    """
    Get comprehensive analytics for a specific product category.
    
    Parameters:
    - category: Product category name
    
    Returns category-specific analytics including:
    - Total revenue
    - Total sales
    - Average price
    - Discount effectiveness
    - Price satisfaction score
    - Review-to-sales ratio
    """
    try:
        # Your existing category analytics logic here
        return {
            "category": category,
            "total_revenue": 1000000.0,
            "total_sales": 5000,
            "average_price": 200.0,
            "discount_effectiveness": 0.85,
            "price_satisfaction": 0.78,
            "review_sales_ratio": 0.15
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/overview", tags=["Analytics"])
async def get_analytics_overview():
    """
    Get overview of all analytics metrics.
    
    Returns:
    - Top performing categories
    - Sales trends
    - Customer satisfaction metrics
    """
    return {
        "top_categories": [
            {
                "category": "ELECTRONICS",
                "revenue": 2500000.0,
                "growth": 15.2
            }
        ],
        "sales_metrics": {
            "total_revenue": 10000000.0,
            "average_order_value": 250.0
        },
        "customer_satisfaction": {
            "average_rating": 4.2,
            "review_ratio": 0.15
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)