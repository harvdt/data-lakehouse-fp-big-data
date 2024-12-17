import os
import json
from datetime import datetime
from kafka import KafkaConsumer
from pyspark.sql import DataFrame
from schemas import bronze_schema
from spark_init import get_spark_session

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'kafka-server'
BATCH_SIZE = 100

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DATA_PATH = os.path.join(BASE_DIR, 'data', 'bronze')
os.makedirs(BRONZE_DATA_PATH, exist_ok=True)


def create_dataframe_from_messages(messages, spark):
    rows = []
    for msg in messages:
        current_time = datetime.now()
        msg['ProcessedTimestamp'] = current_time.isoformat()
        rows.append(msg)
    return spark.createDataFrame(rows, schema=bronze_schema)


def save_batch(batch, spark):
    if not batch:
        return

    try:
        df = create_dataframe_from_messages(batch, spark)
        df.write.format("delta") \
            .mode("append") \
            .option("mergeSchema", "true") \
            .save(BRONZE_DATA_PATH)
    except Exception as e:
        print(f"[CONSUMER] Error saving batch: {str(e)}")


def main():
    spark = get_spark_session()

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='bronze-data-group'
    )

    message_batch = []
    total_processed = 0

    try:
        for message in consumer:
            message_batch.append(message.value)

            if len(message_batch) >= BATCH_SIZE:
                save_batch(message_batch, spark)
                total_processed += len(message_batch)
                message_batch = []

                if total_processed % 1000 == 0:
                    print(f"[CONSUMER] Total records processed: {total_processed}")

    except KeyboardInterrupt:
        if message_batch:
            save_batch(message_batch, spark)
            total_processed += len(message_batch)

        print(f"[CONSUMER] Shutting down. Total records processed: {total_processed}")
        consumer.close()
        spark.stop()


if __name__ == "__main__":
    main()