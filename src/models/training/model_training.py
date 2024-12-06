import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.feature_preprocessor = None
        self.feature_importance = None
        # Add these variables to store train/test data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
    def load_data(self):
        """Load and combine data from gold layer using parquet format"""
        try:
            spark = SparkSession.builder \
                .appName("ModelTraining") \
                .config("spark.sql.legacy.parquet.datetimeRebaseModeInRead", "CORRECTED") \
                .config("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
                .getOrCreate()

            # Set paths
            gold_path = "/home/zaki/kuliah/Bigdata/data-lakehouse-fp-big-data/src/data/gold"
            
            try:
                # Load each table
                revenue_df = spark.read.parquet(f"{gold_path}/revenue_per_category").toPandas()
                discount_df = spark.read.parquet(f"{gold_path}/discount_effectiveness").toPandas()
                price_corr_df = spark.read.parquet(f"{gold_path}/price_rating_correlation").toPandas()
                review_df = spark.read.parquet(f"{gold_path}/review_sales_ratio").toPandas()
                
                logger.info(f"Successfully loaded tables with shapes: ")
                logger.info(f"Revenue: {revenue_df.shape}")
                logger.info(f"Discount: {discount_df.shape}")
                logger.info(f"Price Correlation: {price_corr_df.shape}")
                logger.info(f"Reviews: {review_df.shape}")
                
                # Merge datasets
                merged_df = revenue_df.merge(discount_df, on='Category', suffixes=('', '_disc'))
                merged_df = merged_df.merge(price_corr_df, on='Category', suffixes=('', '_price'))
                merged_df = merged_df.merge(review_df, on='Category', suffixes=('', '_review'))
                
                logger.info(f"Final merged dataset shape: {merged_df.shape}")
                
                logger.info("Available columns in merged dataset:")
                for col in merged_df.columns:
                    logger.info(f"Column: {col}, Type: {merged_df[col].dtype}")
                
                return merged_df
                    
            except Exception as e:
                logger.error(f"Error reading data tables: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            raise
        finally:
            if 'spark' in locals():
                spark.stop()

    def prepare_features(self, df, numeric_features, categorical_features):
        """Prepare features with advanced preprocessing"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse=False))
        ])

        # Combine preprocessors
        self.feature_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return self.feature_preprocessor

    def train_model(self):
        """Train the model with extensive evaluation"""
        logger.info("Starting model training process...")
        
        # Load data
        df = self.load_data()
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Available columns: {df.columns.tolist()}")  # Log available columns

        # Map the features to actual column names from gold layer
        feature_columns = {
            'price': 'AveragePrice',
            'discount': 'AvgDiscountPercentage',
            'rating': 'AvgRating',
            'reviews': 'AvgReviews',
            'sales': 'TotalSales',
            'revenue': 'TotalRevenue',
            'category': 'Category'
        }

        # Prepare features and target
        X = df[[
            feature_columns['price'],
            feature_columns['discount'],
            feature_columns['rating'],
            feature_columns['reviews'],
            feature_columns['category']
        ]]
        y = df[feature_columns['sales']]

        # Split data and store in class variables
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Update feature names
        numeric_features = [
            feature_columns['price'],
            feature_columns['discount'],
            feature_columns['rating'],
            feature_columns['reviews']
        ]
        categorical_features = [feature_columns['category']]

        # Create preprocessing pipeline
        preprocessor = self.prepare_features(df, numeric_features, categorical_features)

        # Try different models
        models = {
            'rf': RandomForestRegressor(random_state=42),
            'gbm': GradientBoostingRegressor(random_state=42),
            'xgb': xgb.XGBRegressor(random_state=42)
        }

        best_score = float('-inf')
        best_model = None
        best_model_name = None

        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Create and train pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            pipeline.fit(self.X_train, self.y_train)
            score = pipeline.score(self.X_test, self.y_test)
            
            logger.info(f"{name} R² score: {score}")
            
            if score > best_score:
                best_score = score
                best_model = pipeline
                best_model_name = name

        logger.info(f"Best model: {best_model_name} with R² score: {best_score}")
        self.model = best_model

        # Generate predictions and store them
        self.y_pred = self.model.predict(self.X_test)
        
        # Evaluate final model
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        logger.info(f"Final Model Metrics:")
        logger.info(f"MSE: {mse}")
        logger.info(f"MAE: {mae}")
        logger.info(f"R² Score: {r2}")

        return self.model

    def save_model(self, filename='sales_predictor.joblib'):
        """Save the trained model"""
        model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metrics': {
                'r2_score': self.model.score(self.X_test, self.y_test),
                'mse': mean_squared_error(self.y_test, self.y_pred),
                'mae': mean_absolute_error(self.y_test, self.y_pred)
            }
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

def main():
    """Main training function"""
    try:
        predictor = SalesPredictor()
        predictor.train_model()
        predictor.save_model()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    
def train_model(self):
    """Train the model with extensive evaluation"""
    logger.info("Starting model training process...")
    
    # Load data
    df = self.load_data()
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Available columns: {df.columns.tolist()}")  # Log available columns

    # Map the features to actual column names from gold layer
    feature_columns = {
        'price': 'AveragePrice',
        'discount': 'AvgDiscountPercentage',
        'rating': 'AvgRating',
        'reviews': 'AvgReviews',
        'sales': 'TotalSales',
        'revenue': 'TotalRevenue',
        'category': 'Category'
    }

    # Prepare features and target
    X = df[[
        feature_columns['price'],
        feature_columns['discount'],
        feature_columns['rating'],
        feature_columns['reviews'],
        feature_columns['category']
    ]]
    y = df[feature_columns['sales']]

    # Split data and store in class variables
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Update feature names in prepare_features method
    numeric_features = [
        feature_columns['price'],
        feature_columns['discount'],
        feature_columns['rating'],
        feature_columns['reviews']
    ]
    categorical_features = [feature_columns['category']]

    # Create preprocessing pipeline
    preprocessor = self.prepare_features(df, numeric_features, categorical_features)

    # Try different models
    models = {
        'rf': RandomForestRegressor(random_state=42),
        'gbm': GradientBoostingRegressor(random_state=42),
        'xgb': xgb.XGBRegressor(random_state=42)
    }

    best_score = float('-inf')
    best_model = None
    best_model_name = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Create and train pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        score = pipeline.score(self.X_test, self.y_test)
        
        logger.info(f"{name} R² score: {score}")
        
        if score > best_score:
            best_score = score
            best_model = pipeline
            best_model_name = name

    logger.info(f"Best model: {best_model_name} with R² score: {best_score}")
    self.model = best_model

    # Generate predictions and store them
    self.y_pred = self.model.predict(self.X_test)
    
    # Evaluate final model
    mse = mean_squared_error(self.y_test, self.y_pred)
    mae = mean_absolute_error(self.y_test, self.y_pred)
    r2 = r2_score(self.y_test, self.y_pred)

    logger.info(f"Final Model Metrics:")
    logger.info(f"MSE: {mse}")
    logger.info(f"MAE: {mae}")
    logger.info(f"R² Score: {r2}")

    return self.model

def prepare_features(self, df, numeric_features, categorical_features):
    """Prepare features with advanced preprocessing"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse=False))
    ])

    # Combine preprocessors
    self.feature_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return self.feature_preprocessor

    def save_model(self, filename='sales_predictor.joblib'):
        """Save the trained model"""
        model_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, filename)
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metrics': {
                'r2_score': self.model.score(self.X_test, self.y_test),
                'mse': mean_squared_error(self.y_test, self.y_pred),
                'mae': mean_absolute_error(self.y_test, self.y_pred)
            }
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

def main():
    """Main training function"""
    try:
        predictor = SalesPredictor()
        predictor.train_model()
        predictor.save_model()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()