import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from delta import *
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.feature_preprocessor = None
        self.target_scaler = None
        self.feature_importance = None
        
    def load_data(self):
        """Load and combine data from gold layer"""
        spark = SparkSession.builder \
            .appName("ModelTraining") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.1.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()

        try:
            # Load data from gold tables
            gold_path = "/home/zaki/kuliah/Bigdata/data-lakehouse-fp-big-data/src/data/gold"
            
            revenue_df = spark.read.format("delta").load(f"{gold_path}/revenue_per_category").toPandas()
            discount_df = spark.read.format("delta").load(f"{gold_path}/discount_effectiveness").toPandas()
            price_corr_df = spark.read.format("delta").load(f"{gold_path}/price_rating_correlation").toPandas()
            review_df = spark.read.format("delta").load(f"{gold_path}/review_sales_ratio").toPandas()
            
            # Merge datasets
            df = revenue_df.merge(discount_df, on='Category', suffixes=('', '_disc'))
            df = df.merge(price_corr_df, on='Category', suffixes=('', '_price'))
            df = df.merge(review_df, on='Category', suffixes=('', '_review'))
            
            return df
            
        finally:
            spark.stop()

    def prepare_features(self, df):
        """Prepare features with advanced preprocessing"""
        # Numeric features
        numeric_features = ['Price', 'Discount', 'Rating', 'NumReviews', 'StockQuantity']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Categorical features
        categorical_features = ['Category']
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

        # Prepare features and target
        X = df[['Price', 'Discount', 'Rating', 'NumReviews', 'StockQuantity', 'Category']]
        y = df['Sales']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create preprocessing pipeline
        preprocessor = self.prepare_features(df)

        # Define models to try
        models = {
            'rf': RandomForestRegressor(random_state=42),
            'gbm': GradientBoostingRegressor(random_state=42),
            'xgb': xgb.XGBRegressor(random_state=42)
        }

        # Parameter grids for each model
        param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            },
            'xgb': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7]
            }
        }

        best_score = float('-inf')
        best_model = None
        best_model_name = None

        # Try each model
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])

            # Grid search
            grid_search = GridSearchCV(
                pipeline,
                {f'regressor__{k}': v for k, v in param_grids[model_name].items()},
                cv=5,
                scoring='r2',
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            score = grid_search.score(X_test, y_test)
            
            logger.info(f"{model_name} R² score: {score}")

            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = model_name

        logger.info(f"Best model: {best_model_name} with R² score: {best_score}")

        # Set the best model
        self.model = best_model

        # Evaluate final model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Final Model Metrics:")
        logger.info(f"MSE: {mse}")
        logger.info(f"MAE: {mae}")
        logger.info(f"R² Score: {r2}")

        # Save feature importance if available
        if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': numeric_features + [f'cat_{cat}' for cat in categorical_features],
                'importance': self.model.named_steps['regressor'].feature_importances_
            }).sort_values('importance', ascending=False)

        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title('Actual vs Predicted Sales')
        plt.savefig('sales_prediction_evaluation.png')
        plt.close()

        return self.model

    def save_model(self, filename='sales_predictor.joblib'):
        """Save the trained model"""
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance
        }, filename)
        logger.info(f"Model saved to {filename}")

    def load_model(self, filename='sales_predictor.joblib'):
        """Load a trained model"""
        loaded = joblib.load(filename)
        self.model = loaded['model']
        self.feature_importance = loaded['feature_importance']
        logger.info(f"Model loaded from {filename}")

def main():
    """Main training function"""
    predictor = SalesPredictor()
    predictor.train_model()
    predictor.save_model()

if __name__ == "__main__":
    main()