#!/bin/bash

SPARK_INIT_SCRIPT="src/data_integration/spark_init.py"
PRODUCER_SCRIPT="src/data_integration/producer.py"
CONSUMER_SCRIPT="src/data_integration/consumer.py"
BRONZE_TO_SILVER_SCRIPT="src/data_processing/bronze_to_silver_processing.py"
SILVER_TO_GOLD_SCRIPT="src/data_processing/silver_to_gold_processing.py"

# Function to check if a process started successfully
check_start() {
    if [ $? -eq 0 ]; then
        echo "$1 started successfully"
    else
        echo "$1 failed to start"
        exit 1
    fi
}

# Initialize Spark session
python3.11 $SPARK_INIT_SCRIPT &
SPARK_PID=$!
check_start "Spark initialization"

# Start the producer
python3.11 $PRODUCER_SCRIPT &
PRODUCER_PID=$!
check_start "Producer"

# Start the consumer
python3.11 $CONSUMER_SCRIPT &
CONSUMER_PID=$!
check_start "Consumer"

wait