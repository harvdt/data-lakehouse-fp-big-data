import os
import json
import time
import csv
from kafka import KafkaProducer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'kafka-server'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_folder = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(dataset_folder, exist_ok=True)

class CSVHandler(FileSystemEventHandler):
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        )

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.csv'):
            print(f"[PRODUCER] Processing file: {event.src_path}")
            time.sleep(1)
            self.process_csv(event.src_path)

    def process_csv(self, filepath):
        try:
            with open(filepath, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)

                for row in csv_reader:
                    try:
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

                        self.producer.send(KAFKA_TOPIC, value=data)
                        # time.sleep(0.1)

                    except Exception as e:
                        print(f"[PRODUCER] Error processing row: {str(e)}")
                        continue

            print(f"[PRODUCER] Finished processing file {filepath}")
            time.sleep(5)

        except Exception as e:
            print(f"[PRODUCER] Error processing file {filepath}: {str(e)}")

def process_existing_files(handler):
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(dataset_folder, filename)
            print(f"[PRODUCER] Processing file: {filepath}")
            handler.process_csv(filepath)

def main():
    event_handler = CSVHandler()

    process_existing_files(event_handler)

    observer = Observer()
    observer.schedule(event_handler, dataset_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()