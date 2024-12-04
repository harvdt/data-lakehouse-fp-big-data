from pyspark.sql.types import *

bronze_schema = StructType([
    StructField("ProductID", IntegerType(), True),
    StructField("ProductName", StringType(), True),
    StructField("Category", StringType(), True),
    StructField("Price", DoubleType(), True),
    StructField("Rating", DoubleType(), True),
    StructField("NumReviews", IntegerType(), True),
    StructField("StockQuantity", IntegerType(), True),
    StructField("Sales", DoubleType(), True),
    StructField("Discount", IntegerType(), True),
    StructField("DateAdded", StringType(), True),
    StructField("City", StringType(), True),
    StructField("StockStatus", BooleanType(), True),
    StructField("ProcessedTimestamp", StringType(), True)
])
