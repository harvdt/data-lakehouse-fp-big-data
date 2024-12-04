import csv
import json
import os
import time
from kafka import KafkaProducer

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'kafka-server'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def send_data(folder):
    dataset_folder = os.path.join(folder, 'data', 'raw')

    for file_name in os.listdir(dataset_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(dataset_folder, file_name)
            print(f'Processing file: {file_name}')

            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)

                for row in csv_reader:
                    stock_status = row[11] == 'TRUE' if len(row) > 11 else False

                    data = {
                        'ProductID': int(row[0]),
                        'ProductName': row[1],
                        'Category': row[2],
                        'Price': float(row[3]),
                        'Rating': float(row[4]),
                        'NumReviews': int(row[5]),
                        'StockQuantity': int(row[6]),
                        'Sales': float(row[7]),
                        'Discount': int(row[8]),
                        'DateAdded': row[9],
                        'City': row[10],
                        'StockStatus': stock_status
                    }

                    producer.send(KAFKA_TOPIC, value=data)
                    print(f"Sent: {data}")
                    # time.sleep(1)

if __name__ == "__main__":
    try:
        send_data(BASE_DIR)
    except KeyboardInterrupt:
        print("Stopping producer...")
    finally:
        producer.close()
