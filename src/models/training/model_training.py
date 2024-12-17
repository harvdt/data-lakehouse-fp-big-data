#!/usr/bin/env python3
"""
E-commerce Sales Prediction Model Training Script
This script trains a stacked ensemble model for predicting e-commerce sales
using data from a Delta Lake gold layer.
"""

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
import os
from delta import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SalesPredictor:
    """A class for training and managing the sales prediction model."""

    def __init__(self):
        """Initialize the SalesPredictor class."""
        self.model = None
        self.feature_preprocessor = None
        self.feature_importance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.categories_ = None
        self.scaler = None

    def load_data(self):
        """Load and combine data from gold layer."""
        spark = None
        try:
            # Initialize Spark with Delta Lake support
            spark = (SparkSession.builder
                .appName("ModelTraining")
                .config("spark.jars.packages", "io.delta:delta-core_2.12:2.1.0")
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g"))

            # Configure Spark with Delta Pip
            spark = configure_spark_with_delta_pip(spark).getOrCreate()

            # Set paths
            #gold_path = "/home/zaki/kuliah/Bigdata/data-lakehouse-fp-big-data/src/data/gold"
            gold_path = "/home/yumx/data-lakehouse-fp-big-data/src/data/gold"

            try:
                # First try reading as Delta format
                revenue_df = spark.read.parquet(f"{gold_path}/revenue_per_category_parquet").toPandas()
                discount_df = spark.read.parquet(f"{gold_path}/discount_effectiveness_parquet").toPandas()
                price_corr_df = spark.read.parquet(f"{gold_path}/price_rating_correlation_parquet").toPandas()
                review_df = spark.read.parquet(f"{gold_path}/review_sales_ratio_parquet").toPandas()

                logger.info("Successfully loaded tables:")
                logger.info(f"Revenue: {revenue_df.shape}")
                logger.info(f"Discount: {discount_df.shape}")
                logger.info(f"Price Correlation: {price_corr_df.shape}")
                logger.info(f"Reviews: {review_df.shape}")

            except Exception as e:
                logger.warning(f"Error reading as Delta format: {e}")
                logger.info("Trying to read as parquet format...")

                # Fallback to reading as plain parquet
                revenue_df = pd.read_parquet(f"{gold_path}/revenue_per_category_parquet")
                discount_df = pd.read_parquet(f"{gold_path}/discount_effectiveness_parquet")
                price_corr_df = pd.read_parquet(f"{gold_path}/price_rating_correlation_parquet")
                review_df = pd.read_parquet(f"{gold_path}/review_sales_ratio_parquet")

                logger.info("Successfully loaded tables using pandas:")
                logger.info(f"Revenue: {revenue_df.shape}")
                logger.info(f"Discount: {discount_df.shape}")
                logger.info(f"Price Correlation: {price_corr_df.shape}")
                logger.info(f"Reviews: {review_df.shape}")

            # Store unique categories
            self.categories_ = sorted(revenue_df['Category'].unique())
            logger.info(f"Found categories: {self.categories_}")

            # Merge datasets
            merged_df = revenue_df.merge(discount_df, on='Category', suffixes=('', '_disc'))
            merged_df = merged_df.merge(price_corr_df, on='Category', suffixes=('', '_price'))
            merged_df = merged_df.merge(review_df, on='Category', suffixes=('', '_review'))

            logger.info(f"Final merged dataset shape: {merged_df.shape}")
            logger.info("Available columns: %s", merged_df.columns.tolist())

            # Handle missing values
            if merged_df.isnull().sum().sum() > 0:
                logger.warning("Found null values in merged dataset")
                merged_df = merged_df.fillna(0)

            return merged_df

        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            raise
        finally:
            if spark:
                spark.stop()

    def create_advanced_features(self, df):
        """Create advanced features for prediction."""
        df = df.copy()

        try:
            # Price-based features
            df['PriceToAvgRatio'] = df['AveragePrice'] / df['AveragePrice'].mean()
            df['PriceVariance'] = np.abs(df['AveragePrice'] - df['AveragePrice'].mean()) / df['AveragePrice'].std()
            df['PriceCategory'] = pd.qcut(df['AveragePrice'], q=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])

            # Discount features
            df['DiscountEfficiency'] = np.where(
                df['RevenueLossFromDiscount'] > 0,
                df['TotalDiscountedRevenue'] / df['RevenueLossFromDiscount'],
                0
            )
            df['DiscountROICategory'] = pd.qcut(df['DiscountROI'].clip(lower=0), q=3, labels=['Low', 'Medium', 'High'])
            df['DiscountEffectiveness'] = df['TotalDiscountedRevenue'] / df['PotentialRevenue']

            # Engagement features
            df['ReviewsPerSale'] = df['TotalReviews'] / df['TotalSales'].replace(0, 1)
            df['HighRatingRatio'] = df['HighRatingCount'] / df['ProductCount'].replace(0, 1)
            df['EngagementScore'] = (df['ReviewsPerSale'] * df['HighRatingRatio'] * df['AvgRating'])
            df['CustomerSatisfaction'] = (df['AvgRating'] * df['HighRatingRatio']) / (df['DiscountEffectiveness'] + 1)

            # Revenue and sales metrics
            df['RevenuePerProduct'] = df['TotalRevenue'] / df['ProductCount']
            df['SalesEfficiency'] = df['TotalSales'] / (df['ProductCount'] * df['AvgDiscountPercentage'] + 1)

            # Interaction features
            df['PriceRatingInteraction'] = df['AveragePrice'] * df['AvgRating']
            df['DiscountRatingInteraction'] = df['AvgDiscountPercentage'] * df['AvgRating']
            df['PriceDiscountInteraction'] = df['AveragePrice'] * df['AvgDiscountPercentage']

            # Performance metrics
            df['MarketPerformance'] = df['TotalRevenue'] / (df['ProductCount'] * df['AveragePrice'])
            df['CustomerEngagementIndex'] = (df['TotalReviews'] * df['AvgRating']) / df['TotalSales']
            df['ProfitabilityScore'] = df['TotalRevenue'] * (1 - df['AvgDiscountPercentage']/100)

            # Normalized features
            for col in ['TotalRevenue', 'TotalSales', 'AveragePrice']:
                df[f'Normalized_{col}'] = (df[col] - df[col].mean()) / df[col].std()

            return df

        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise

    def prepare_advanced_features(self, df, numeric_features, categorical_features):
        """Prepare features for modeling with advanced preprocessing."""
        numeric_transformer = Pipeline(steps=[
            ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.feature_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return self.feature_preprocessor

    def create_stacked_model(self):
        """Create a stacked ensemble model."""
        estimators = [
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )),
            ('gbm', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )),
            ('xgb', xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.01,
                max_depth=3,
                random_state=42
            ))
        ]

        final_estimator = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1
        )

    def extract_feature_importance(self, model, numeric_features, categorical_features):
        """Extract feature importance from the model safely."""
        try:
            final_estimator = model.named_steps['regressor'].final_estimator_
            if not hasattr(final_estimator, 'feature_importances_'):
                logger.warning("Final estimator doesn't have feature importance attribute")
                return None

            # Get transformed feature names
            transformed_features = []
            transformed_features.extend(numeric_features)

            # Add encoded categorical feature names
            for feature in categorical_features:
                if feature == 'Category':
                    values = sorted(self.categories_)
                elif feature == 'PriceCategory':
                    values = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
                elif feature == 'DiscountROICategory':
                    values = ['Low', 'Medium', 'High']
                else:
                    continue

                transformed_features.extend([f"{feature}_{val}" for val in values[1:]])

            importances = final_estimator.feature_importances_

            if len(importances) != len(transformed_features):
                logger.warning(f"Feature importance length mismatch: {len(importances)} vs {len(transformed_features)}")
                min_len = min(len(importances), len(transformed_features))
                importances = importances[:min_len]
                transformed_features = transformed_features[:min_len]

            importance_df = pd.DataFrame({
                'feature': transformed_features,
                'importance': importances
            })

            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df['importance_percentage'] = importance_df['importance'] * 100

            return importance_df

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return None

    def train_model(self):
        """Train the model with all improvements."""
        logger.info("Starting enhanced model training process...")

        try:
            # Load and prepare data
            df = self.load_data()
            df = self.create_advanced_features(df)
            logger.info(f"Created advanced features. Shape: {df.shape}")

            # Define features
            numeric_features = [
                'AveragePrice', 'AvgDiscountPercentage', 'AvgRating', 'AvgReviews',
                'PriceToAvgRatio', 'PriceVariance', 'DiscountEfficiency',
                'DiscountEffectiveness', 'ReviewsPerSale', 'HighRatingRatio',
                'EngagementScore', 'CustomerSatisfaction', 'RevenuePerProduct',
                'SalesEfficiency', 'PriceRatingInteraction', 'DiscountRatingInteraction',
                'PriceDiscountInteraction', 'MarketPerformance',
                'CustomerEngagementIndex', 'ProfitabilityScore',
                'Normalized_TotalRevenue', 'Normalized_TotalSales',
                'Normalized_AveragePrice'
            ]
            categorical_features = ['Category', 'PriceCategory', 'DiscountROICategory']

            logger.info(f"Number of numeric features: {len(numeric_features)}")
            logger.info(f"Number of categorical features: {len(categorical_features)}")

            # Prepare features and target
            X = df[numeric_features + categorical_features]
            y = df['TotalSales']

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Create and train model
            preprocessor = self.prepare_advanced_features(df, numeric_features, categorical_features)
            stacked_model = self.create_stacked_model()

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', stacked_model)
            ])

            logger.info("Training stacked model...")
            pipeline.fit(self.X_train, self.y_train)

            # Evaluate model
            self.y_pred = pipeline.predict(self.X_test)

            r2 = r2_score(self.y_test, self.y_pred)
            mse = mean_squared_error(self.y_test, self.y_pred)
            mae = mean_absolute_error(self.y_test, self.y_pred)
            rmse = np.sqrt(mse)

            logger.info("Model Performance:")
            logger.info(f"R² score: {r2:.4f}")
            logger.info(f"MSE: {mse:.4f}")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")

            self.model = pipeline

            # Extract feature importance
            self.feature_importance = self.extract_feature_importance(
                self.model, numeric_features, categorical_features
            )

            if self.feature_importance is not None:
                logger.info("\nTop 10 Most Important Features:")
                logger.info(self.feature_importance.head(10))

            return self.model

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def save_model(self, filename='sales_predictor.joblib'):
        """Save the trained model"""
        try:
            model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, filename)

            model_data = {
                'model': self.model,
                'feature_importance': self.feature_importance,
                'categories': self.categories_,
                'metrics': {
                    'r2_score': r2_score(self.y_test, self.y_pred),
                    'mse': mean_squared_error(self.y_test, self.y_pred),
                    'mae': mean_absolute_error(self.y_test, self.y_pred)
                }
            }

            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    """Main training function"""
    try:
        predictor = SalesPredictor()
        predictor.train_model()
        predictor.save_model()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()