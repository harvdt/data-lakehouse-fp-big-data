import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, to_date, current_timestamp, 
    round, upper, trim, regexp_replace,
    year, month, dayofmonth, count,
    avg, min, max
)
from pyspark.sql.types import (
    StructType, StructField, IntegerType, 
    StringType, DoubleType, BooleanType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from delta import *

BRONZE_DATA_PATH = "/home/aveee/data-lakehouse-fp-big-data/src/data/bronze"
#BRONZE_DATA_PATH = "/home/zaki/kuliah/Bigdata/data-lakehouse-fp-big-data/src/data/bronze"
DATA_DIR = os.path.dirname(BRONZE_DATA_PATH)
SILVER_DATA_PATH = os.path.join(DATA_DIR, "silver")

def get_spark_session():
    """Create a new Spark session with Delta Lake support"""
    builder = SparkSession.builder \
        .appName("XYZEcommerceData") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.memory.fraction", "0.7") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g")

    return configure_spark_with_delta_pip(builder).getOrCreate()

def clean_and_validate_data(spark, bronze_df):
    """
    Clean and validate the bronze data with fixed sales and discount handling
    """
    return bronze_df \
        .dropDuplicates(['ProductID']) \
        .withColumn('Price', 
            col('Price').cast('float')) \
        .withColumn('Rating',
            col('Rating').cast('float')) \
        .withColumn('NumReviews',
            col('NumReviews').cast('int')) \
        .withColumn('StockQuantity',
            col('StockQuantity').cast('int')) \
        .withColumn('Sales',
            when(col('Sales') < 0, 0.0)
            .when(col('Sales') <= 1.0, col('Sales') * 1000)  # Scale up small sales values
            .otherwise(col('Sales'))) \
        .withColumn('Discount',
            when(col('Discount') > 100, col('Discount') / 100)  # Convert large discount values to percentage
            .when(col('Discount') < 0, 0)
            .when(col('Discount') > 50, 50)
            .otherwise(col('Discount'))) \
        .withColumn('DateAdded', to_date(col('DateAdded'))) \
        .withColumn('ProductName', trim(col('ProductName'))) \
        .withColumn('Category', upper(trim(col('Category')))) \
        .withColumn('City', trim(regexp_replace(col('City'), '[^a-zA-Z\\s]', ''))) \
        .withColumn('Price',
            when(col('Price') < 0, None)
            .when(col('Price') > 500, None)  
            .otherwise(round(col('Price'), 2))) \
        .withColumn('Rating',
            when(col('Rating') < 1.0, None)
            .when(col('Rating') > 5.0, None)
            .otherwise(col('Rating'))) \
        .withColumn('NumReviews',
            when(col('NumReviews') < 0, 0)
            .when(col('NumReviews') > 5000, None)  
            .otherwise(col('NumReviews'))) \
        .withColumn('StockQuantity',
            when(col('StockQuantity') < 0, 0)
            .when(col('StockQuantity') > 1000, None)  
            .otherwise(col('StockQuantity'))) \
        .withColumn('Year', year(col('DateAdded'))) \
        .withColumn('Month', month(col('DateAdded')))
        
def standardize_numeric_features(df):
    """
    Standardize numeric features based on analysis:
    - Price
    - Rating
    - NumReviews
    - StockQuantity
    - Sales
    """
    numeric_cols = ['Price', 'Rating', 'NumReviews', 'StockQuantity', 'Sales']
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol='features')
    scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
    pipeline = Pipeline(stages=[assembler, scaler])
    
    return pipeline.fit(df).transform(df)

def calculate_data_quality_metrics(df):
    """
    Calculate and log data quality metrics
    """
    total_records = df.count()
    null_counts = df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]).collect()[0]
    
    print(f"Total Records: {total_records}")
    print("\nNull Counts per Column:")
    for col_name in df.columns:
        print(f"{col_name}: {getattr(null_counts, col_name)}")

def process_bronze_to_silver():
    """
    Main function to process bronze data into silver data with validation
    """
    try:
        print(f"Bronze data path: {BRONZE_DATA_PATH}")
        print(f"Silver data path: {SILVER_DATA_PATH}")
        
        spark = get_spark_session()
        
        # Read bronze data
        bronze_df = spark.read.format("delta").load(BRONZE_DATA_PATH)
        
        # Show sales data before transformation
        print("\nBronze Data Sales Sample:")
        bronze_df.select("Category", "Sales", "Price", "Discount").show(5)
        print("\nBronze Sales Statistics:")
        bronze_df.select(
            round(avg("Sales"), 2).alias("avg_sales"),
            min("Sales").alias("min_sales"),
            max("Sales").alias("max_sales")
        ).show()
        
        # Clean and validate
        cleaned_df = clean_and_validate_data(spark, bronze_df)
        
        # Standardize numeric features
        silver_df = standardize_numeric_features(cleaned_df)
        
        # Show sales data after transformation
        print("\nSilver Data Sales Sample:")
        silver_df.select("Category", "Sales", "Price", "Discount").show(5)
        print("\nSilver Sales Statistics:")
        silver_df.select(
            round(avg("Sales"), 2).alias("avg_sales"),
            min("Sales").alias("min_sales"),
            max("Sales").alias("max_sales")
        ).show()
        
        # Calculate quality metrics before saving
        calculate_data_quality_metrics(silver_df)
        
        # Write to silver layer
        silver_df.write \
            .format("delta") \
            .mode("overwrite") \
            .partitionBy("Year", "Month") \
            .option("overwriteSchema", "true") \
            .save(SILVER_DATA_PATH)
        
        print("Successfully processed bronze data to silver layer")
        
    except Exception as e:
        print(f"Error processing bronze to silver: {e}")
        raise
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    process_bronze_to_silver()