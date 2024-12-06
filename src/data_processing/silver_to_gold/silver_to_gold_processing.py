import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, avg, round, count,
    corr, when, expr, row_number, lit
)
from pyspark.sql.window import Window
from delta import *

# Configuration
SILVER_DATA_PATH = "/home/zaki/kuliah/Bigdata/data-lakehouse-fp-big-data/src/data/silver"
DATA_DIR = os.path.dirname(SILVER_DATA_PATH)
GOLD_DATA_PATH = os.path.join(DATA_DIR, "gold")

def get_spark_session():
    """Create a new Spark session with Delta Lake support"""
    builder = SparkSession.builder \
        .appName("XYZEcommerceDataGold") \
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g")

    return configure_spark_with_delta_pip(builder).getOrCreate()

def calculate_sales_performance(df):
    """Calculate sales performance metrics with corrected calculations"""
    # Calculate revenue properly
    df_with_revenue = df.withColumn(
        "Revenue", 
        round(col("Price") * col("Sales") * (1.0 - col("Discount")/100), 2)
    )

    # Revenue per category
    revenue_per_category = df_with_revenue.groupBy("Category").agg(
        round(sum("Revenue"), 2).alias("TotalRevenue"),
        sum("Sales").alias("TotalSales"),
        round(avg("Price"), 2).alias("AveragePrice"),
        count("*").alias("ProductCount"),
        round(avg("Discount"), 2).alias("AvgDiscountPercentage")
    ).orderBy(col("TotalRevenue").desc())

    # Discount impact
    discount_impact = df.groupBy("Category").agg(
        round(avg("Discount"), 2).alias("AvgDiscountPercentage"),
        sum("Sales").alias("TotalSales"),
        round(avg(col("Sales")), 2).alias("AvgSalesWithDiscount"),
        round(avg(when(col("Discount") == 0, col("Sales"))), 2).alias("AvgSalesWithoutDiscount")
    ).withColumn(
        "DiscountImpactPercentage",
        when(
            col("AvgSalesWithoutDiscount") > 0,
            round((col("AvgSalesWithDiscount") - col("AvgSalesWithoutDiscount")) / 
                  col("AvgSalesWithoutDiscount") * 100, 2)
        ).otherwise(lit(0.0))
    )

    return revenue_per_category, discount_impact

def analyze_pricing_strategy(df):
    """Analyze pricing strategy"""
    # Discount effectiveness
    discount_effectiveness = df.withColumn(
        "DiscountedRevenue", 
        round(col("Price") * col("Sales") * (1.0 - col("Discount")/100), 2)
    ).withColumn(
        "FullPriceRevenue",
        round(col("Price") * col("Sales"), 2)
    ).groupBy("Category").agg(
        round(avg("Discount"), 2).alias("AvgDiscountPercentage"),
        round(sum("DiscountedRevenue"), 2).alias("TotalDiscountedRevenue"),
        round(sum("FullPriceRevenue"), 2).alias("PotentialRevenue"),
        round(avg("Sales"), 2).alias("AvgSales")
    ).withColumn(
        "RevenueLossFromDiscount",
        round(col("PotentialRevenue") - col("TotalDiscountedRevenue"), 2)
    ).withColumn(
        "DiscountROI",
        when(
            col("RevenueLossFromDiscount") > 0,
            round(col("TotalDiscountedRevenue") / col("RevenueLossFromDiscount"), 3)
        ).otherwise(0.0)
    )

    # Price rating correlation
    price_rating_corr = df.groupBy("Category").agg(
        round(corr("Price", "Rating"), 3).alias("PriceRatingCorrelation"),
        round(avg("Price"), 2).alias("AvgPrice"),
        round(avg("Rating"), 2).alias("AvgRating"),
        round(avg("NumReviews"), 0).alias("AvgReviews")
    ).withColumn(
        "PriceRatingRelationship",
        when(col("PriceRatingCorrelation") > 0.3, "Strong Positive")
        .when(col("PriceRatingCorrelation") < -0.3, "Strong Negative")
        .otherwise("Weak")
    )

    return discount_effectiveness, price_rating_corr

def calculate_customer_satisfaction(df):
    """Calculate customer satisfaction metrics"""
    avg_price = df.agg(avg("Price")).first()[0]

    # Price satisfaction analysis
    price_satisfaction = df.withColumn(
        "PriceRange",
        when(col("Price") < avg_price * 0.5, "Budget")
        .when(col("Price") < avg_price * 1.0, "Economy")
        .when(col("Price") < avg_price * 1.5, "Premium")
        .otherwise("Luxury")
    ).groupBy("PriceRange", "Category").agg(
        round(avg("Rating"), 2).alias("AvgRating"),
        round(avg("Price"), 2).alias("AvgPrice"),
        sum(when(col("Sales").isNotNull(), col("Sales")).otherwise(lit(0))).alias("TotalSales"),
        count("*").alias("ProductCount"),
        round(avg("NumReviews"), 0).alias("AvgReviews")
    ).orderBy("Category", "PriceRange")

    # Review-to-sales ratio
    review_sales_ratio = df.groupBy("Category").agg(
        sum("NumReviews").alias("TotalReviews"),
        sum(when(col("Sales").isNotNull(), col("Sales")).otherwise(lit(0))).alias("TotalSales"),
        round(avg("Rating"), 2).alias("AvgRating"),
        count(when(col("Rating") >= 4.0, True)).alias("HighRatingCount")
    ).withColumn(
        "ReviewToSalesRatio",
        when(col("TotalSales") > 0, 
             round(col("TotalReviews") / col("TotalSales"), 4))
        .otherwise(lit(0.0))
    ).withColumn(
        "CustomerEngagementLevel",
        when(col("ReviewToSalesRatio") > 0.5, "High")
        .when(col("ReviewToSalesRatio") > 0.2, "Medium")
        .otherwise("Low")
    )

    return price_satisfaction, review_sales_ratio

def save_analytics(data, name, path):
    """Save analytics in both Delta and Parquet format"""
    # Save as Delta
    delta_path = os.path.join(path, name)
    data.write \
        .format("delta") \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .option("overwriteSchema", "true") \
        .save(delta_path)
    
    # Save as Parquet
    parquet_path = os.path.join(path, f"{name}_parquet")
    data.write \
        .mode("overwrite") \
        .parquet(parquet_path)
    
    print(f"Saved {name} to:")
    print(f"  Delta: {delta_path}")
    print(f"  Parquet: {parquet_path}")

def process_silver_to_gold():
    """Main function with enhanced error handling and validation"""
    spark = None
    try:
        print(f"Starting silver to gold transformation...")
        print(f"Reading from: {SILVER_DATA_PATH}")
        print(f"Writing to: {GOLD_DATA_PATH}")
        
        os.makedirs(GOLD_DATA_PATH, exist_ok=True)
        spark = get_spark_session()
        
        # Read and validate silver data
        silver_df = spark.read.format("delta").load(SILVER_DATA_PATH)
        
        # Validate data
        total_records = silver_df.count()
        null_counts = {column_name: silver_df.filter(col(column_name).isNull()).count() 
                      for column_name in silver_df.columns}
        
        print(f"\nData Validation:")
        print(f"Total Records: {total_records}")
        print("Null counts per column:")
        for column_name, count in null_counts.items():
            print(f"{column_name}: {count}")
            
        print("\nSales Data Sample:")
        silver_df.select("Category", "Sales", "Price", "Discount").show(5)
        
        # Calculate all metrics
        revenue_per_category, discount_impact = calculate_sales_performance(silver_df)
        discount_effectiveness, price_rating_corr = analyze_pricing_strategy(silver_df)
        price_satisfaction, review_sales_ratio = calculate_customer_satisfaction(silver_df)
        
        # Save all analytics using the save_analytics function
        analytics = {
            "revenue_per_category": revenue_per_category,
            "discount_impact": discount_impact,
            "discount_effectiveness": discount_effectiveness,
            "price_rating_correlation": price_rating_corr,
            "price_satisfaction": price_satisfaction,
            "review_sales_ratio": review_sales_ratio
        }
        
        for name, data in analytics.items():
            save_analytics(data, name, GOLD_DATA_PATH)
            print(f"\nSample data for {name}:")
            data.show(3, truncate=False)
        
        print("\nSuccessfully processed silver data to gold layer")
        
    except Exception as e:
        print(f"Error processing silver to gold: {e}")
        raise
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    process_silver_to_gold()