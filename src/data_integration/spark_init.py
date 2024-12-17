from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip  # Import the function

def get_spark_session():
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
