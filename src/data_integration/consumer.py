import json
import os
import sys
import time
from kafka import KafkaConsumer
from schemas import bronze_schema
from spark_init import get_spark_session
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'kafka-server'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'bronze')

os.makedirs(BRONZE_DATA_PATH, exist_ok=True)

spark = get_spark_session()

def create_dataframe_from_messages(messages):
    rows = []
    for msg in messages:
        current_time = datetime.now()
        msg['ProcessedTimestamp'] = current_time.isoformat()
        rows.append(msg)

    return spark.createDataFrame(rows, schema=bronze_schema)

def main():
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='bronze-data-group'
    )

    batch_size = 100
    current_batch = []
    last_save_time = time.time()

    try:
        for message in consumer:
            data = message.value

            try:
                current_batch.append(data)
                current_time = time.time()

                print(f"Processed message ID: {data['ProductID']}")

                if len(current_batch) >= batch_size or (current_time - last_save_time) >= 60:
                    df = create_dataframe_from_messages(current_batch)

                    df.write.format("delta") \
                        .mode("append") \
                        .option("mergeSchema", "true") \
                        .save(BRONZE_DATA_PATH)

                    current_batch = []
                    last_save_time = current_time

            except Exception as e:
                print(f"Error processing message with ID {data['ProductID']}: {e}")
                continue

    except KeyboardInterrupt:
        print("Stopping consumer gracefully...")

    finally:
        if current_batch:
            try:
                print(f"Saving final batch of {len(current_batch)} records...")

                df = create_dataframe_from_messages(current_batch)

                df.write.format("delta") \
                    .mode("append") \
                    .option("mergeSchema", "true") \
                    .save(BRONZE_DATA_PATH)

                print(f"Successfully saved final batch of {len(current_batch)} records")
            except Exception as e:
                print(f"Error saving final batch: {e}")

        print("Closing consumer and Spark session...")
        consumer.close()
        spark.stop()

if __name__ == "__main__":
    main()
