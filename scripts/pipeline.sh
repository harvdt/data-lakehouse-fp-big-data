#!/bin/bash

SPARK_INIT_SCRIPT="src/data_integration/spark_init.py"
PRODUCER_SCRIPT="src/data_integration/producer.py"
CONSUMER_SCRIPT="src/data_integration/consumer.py"

echo "Initializing spark session..."
python3.11 $SPARK_INIT_SCRIPT &
SPARK_INIT_PID=$!

sleep 1

echo "Starting producer..."
python3.11 $PRODUCER_SCRIPT &
PRODUCER_PID=$!

echo "Starting consumer..."
python3.11 $CONSUMER_SCRIPT &
CONSUMER_PID=$!

wait $SPARK_INIT_PID
wait $PRODUCER_PID
wait $CONSUMER_PID

echo "Pipeline execution completed."
