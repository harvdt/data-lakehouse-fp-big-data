import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from delta import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, count, round

app = FastAPI()

# Paths
GOLD_PATH = "/home/zaki/kuliah/Bigdata/data-lakehouse-fp-big-data/src/data/gold"

class ProductData(BaseModel):
    Category: str
    Price: float
    Discount: float
    Rating: float
    NumReviews: int
    StockQuantity: int

def get_spark_session():
    return SparkSession.builder \
        .appName("Analytics API") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

@app.get("/analytics/category/{category}")
async def get_category_analytics(category: str):
    """Get analytics for a specific category from gold layer data"""
    try:
        spark = get_spark_session()
        
        # Load data from gold tables
        revenue_df = spark.read.format("delta").load(f"{GOLD_PATH}/revenue_per_category")
        discount_df = spark.read.format("delta").load(f"{GOLD_PATH}/discount_effectiveness")
        price_corr_df = spark.read.format("delta").load(f"{GOLD_PATH}/price_rating_correlation")
        review_ratio_df = spark.read.format("delta").load(f"{GOLD_PATH}/review_sales_ratio")
        
        # Filter for specific category
        category_revenue = revenue_df.filter(col("Category") == category).first()
        category_discount = discount_df.filter(col("Category") == category).first()
        category_price_corr = price_corr_df.filter(col("Category") == category).first()
        category_review = review_ratio_df.filter(col("Category") == category).first()
        
        return {
            "sales_performance": {
                "total_revenue": float(category_revenue["TotalRevenue"]),
                "total_sales": float(category_revenue["TotalSales"]),
                "avg_price": float(category_revenue["AveragePrice"])
            },
            "pricing_strategy": {
                "discount_effectiveness": float(category_discount["DiscountROI"]),
                "price_rating_correlation": float(category_price_corr["PriceRatingCorrelation"])
            },
            "customer_satisfaction": {
                "price_satisfaction": float(category_price_corr["AvgRating"]),
                "review_sales_ratio": float(category_review["ReviewToSalesRatio"])
            }
        }
    finally:
        spark.stop()

@app.get("/analytics/overview")
async def get_analytics_overview():
    """Get overview analytics from gold layer data"""
    try:
        spark = get_spark_session()
        
        # Load and process data from gold tables
        revenue_df = spark.read.format("delta").load(f"{GOLD_PATH}/revenue_per_category")
        discount_df = spark.read.format("delta").load(f"{GOLD_PATH}/discount_effectiveness")
        satisfaction_df = spark.read.format("delta").load(f"{GOLD_PATH}/price_rating_correlation")
        
        # Get top categories by revenue
        top_categories = revenue_df.orderBy(col("TotalRevenue").desc()).limit(5).collect()
        
        # Get best performing discount category
        best_discount = discount_df.orderBy(col("DiscountROI").desc()).first()
        
        # Get highest rated category
        highest_rated = satisfaction_df.orderBy(col("AvgRating").desc()).first()
        
        return {
            "top_categories": [
                {
                    "category": cat["Category"],
                    "revenue": float(cat["TotalRevenue"]),
                    "sales": float(cat["TotalSales"])
                } for cat in top_categories
            ],
            "discount_effectiveness": {
                "best_performing": {
                    "category": best_discount["Category"],
                    "roi": float(best_discount["DiscountROI"])
                }
            },
            "customer_satisfaction": {
                "highest_rated": {
                    "category": highest_rated["Category"],
                    "rating": float(highest_rated["AvgRating"])
                }
            }
        }
    finally:
        spark.stop()

@app.post("/predict/sales_performance")
async def predict_sales_performance(product: ProductData):
    """Predict sales performance using the trained model"""
    try:
        spark = get_spark_session()
        
        # Load historical data for the category
        revenue_df = spark.read.format("delta").load(f"{GOLD_PATH}/revenue_per_category")
        category_data = revenue_df.filter(col("Category") == product.Category).first()
        
        # Calculate predictions based on historical data and product features
        avg_sales = float(category_data["TotalSales"]) / float(category_data["ProductCount"])
        predicted_sales = avg_sales * (1 + (product.Rating - float(category_data["AvgDiscountPercentage"]) / 100))
        predicted_revenue = predicted_sales * product.Price * (1 - product.Discount/100)
        
        return {
            "predicted_sales": float(predicted_sales),
            "predicted_revenue": float(predicted_revenue),
            "discount_impact": float(predicted_sales * product.Discount/100)
        }
    finally:
        spark.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)