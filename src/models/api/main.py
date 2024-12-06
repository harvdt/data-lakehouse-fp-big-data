from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Union
import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from enum import Enum
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api.log'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="XYZ E-commerce Analytics API",
    description="API for e-commerce analytics and sales predictions",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for paths
BASE_PATH = "/home/zaki/kuliah/Bigdata/data-lakehouse-fp-big-data"
GOLD_PATH = os.path.join(BASE_PATH, "src/data/gold")
MODEL_PATH = os.path.join(BASE_PATH, "src/models/training/trained_models/sales_predictor.joblib")

# Initialize Spark Session with Delta
def get_spark_session():
    try:
        builder = SparkSession.builder \
            .appName("EcommerceAPI") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.1.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g")

        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        return spark
    except Exception as e:
        logger.error(f"Error creating Spark session: {e}")
        raise RuntimeError(f"Failed to initialize Spark: {str(e)}")

# Helper function to read parquet data instead of Delta
def read_analytics_data(table_name: str) -> pd.DataFrame:
    try:
        parquet_path = os.path.join(GOLD_PATH, f"{table_name}_parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Data not found at {parquet_path}")
        
        spark = get_spark_session()
        df = spark.read.parquet(parquet_path)
        pdf = df.toPandas()
        spark.stop()
        return pdf
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data for {table_name} not found"
        )
    except Exception as e:
        logger.error(f"Error reading data for {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading {table_name} data: {str(e)}"
        )

# Load ML model
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    logger.info("ML model loaded successfully")
except Exception as e:
    logger.error(f"Error loading ML model: {e}")
    model = None

# Pydantic models
class PredictionRequest(BaseModel):
    price: float = Field(..., gt=0, description="Product price")
    discount_percentage: float = Field(..., ge=0, le=100, description="Discount percentage")
    rating: float = Field(..., ge=1, le=5, description="Product rating")
    num_reviews: int = Field(..., ge=0, description="Number of reviews")
    category: str = Field(..., description="Product category")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "price": 99.99,
                "discount_percentage": 15,
                "rating": 4.5,
                "num_reviews": 100,
                "category": "Electronics"
            }
        }
    )

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "api_name": "XYZ E-commerce Analytics API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/categories")
async def get_categories():
    """Get list of available categories"""
    try:
        df = read_analytics_data("revenue_per_category")
        categories = sorted(df['Category'].unique().tolist())
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching categories: {str(e)}"
        )

@app.get("/analytics/revenue")
async def get_revenue_analytics(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get revenue analytics per category"""
    try:
        df = read_analytics_data("revenue_per_category")
        if category:
            if category not in df['Category'].values:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category: {category}"
                )
            df = df[df['Category'] == category]
        return JSONResponse(content=df.to_dict('records'))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in revenue analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching revenue analytics: {str(e)}"
        )

@app.get("/analytics/discount")
async def get_discount_analytics(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get discount effectiveness analytics"""
    try:
        df = read_analytics_data("discount_effectiveness")
        if category:
            if category not in df['Category'].values:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category: {category}"
                )
            df = df[df['Category'] == category]
        return JSONResponse(content=df.to_dict('records'))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in discount analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching discount analytics: {str(e)}"
        )

@app.get("/analytics/price-rating")
async def get_price_rating_analytics(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get price-rating correlation analytics"""
    try:
        df = read_analytics_data("price_rating_correlation")
        if category:
            if category not in df['Category'].values:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category: {category}"
                )
            df = df[df['Category'] == category]
        return JSONResponse(content=df.to_dict('records'))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in price-rating analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching price-rating analytics: {str(e)}"
        )

@app.get("/analytics/customer-satisfaction")
async def get_customer_satisfaction_analytics(
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get customer satisfaction metrics"""
    try:
        df = read_analytics_data("review_sales_ratio")
        if category:
            if category not in df['Category'].values:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category: {category}"
                )
            df = df[df['Category'] == category]
        return JSONResponse(content=df.to_dict('records'))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in customer satisfaction analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching customer satisfaction analytics: {str(e)}"
        )
        
@app.post("/predict/sales", response_model=Dict[str, float])
async def predict_sales(request: PredictionRequest):
    """Predict sales based on product features"""
    try:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model not available"
            )

        # Create base features
        input_data = pd.DataFrame([{
            'AveragePrice': request.price,
            'AvgDiscountPercentage': request.discount_percentage,
            'AvgRating': request.rating,
            'AvgReviews': request.num_reviews,
            'Category': request.category,
        }])

        # Determine price category based on fixed thresholds
        price_thresholds = {
            'VeryLow': 100,
            'Low': 200,
            'Medium': 300,
            'High': 400,
            'VeryHigh': float('inf')
        }
        
        price_category = 'VeryHigh'
        for category, threshold in price_thresholds.items():
            if request.price <= threshold:
                price_category = category
                break

        # Create all required features
        features = {
            'PriceToAvgRatio': 1.0,  # Normalized assumption
            'PriceVariance': 0.0,    # Normalized assumption
            'DiscountEfficiency': request.discount_percentage / 100,
            'DiscountEffectiveness': 1 - (request.discount_percentage / 100),
            'ReviewsPerSale': request.num_reviews / 100,  # Normalized assumption
            'HighRatingRatio': 1 if request.rating >= 4 else 0,
            'EngagementScore': request.rating * (request.num_reviews / 100),
            'CustomerSatisfaction': request.rating * (1 - request.discount_percentage/100),
            'RevenuePerProduct': request.price * (1 - request.discount_percentage/100),
            'SalesEfficiency': 1.0,  # Default for single prediction
            'PriceRatingInteraction': request.price * request.rating,
            'DiscountRatingInteraction': request.discount_percentage * request.rating,
            'PriceDiscountInteraction': request.price * request.discount_percentage,
            'MarketPerformance': 1.0,  # Default for single prediction
            'CustomerEngagementIndex': request.num_reviews * request.rating,
            'ProfitabilityScore': request.price * (1 - request.discount_percentage/100),
            'Normalized_TotalRevenue': 0.0,  # Will be scaled by the model
            'Normalized_TotalSales': 0.0,    # Will be scaled by the model
            'Normalized_AveragePrice': 0.0,  # Will be scaled by the model
            'PriceCategory': price_category,
            'DiscountROICategory': 'Medium'  # Default category
        }

        # Update input data with all features
        for feature, value in features.items():
            input_data[feature] = value

        # Make prediction
        prediction = model_data['model'].predict(input_data)[0]
        confidence = model_data['metrics']['r2_score']

        # Return prediction with additional context
        return {
            "predicted_sales": float(prediction),
            "confidence_score": float(confidence)
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)