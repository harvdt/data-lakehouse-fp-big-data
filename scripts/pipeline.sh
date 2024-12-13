#!/bin/bash

SPARK_INIT_SCRIPT="src/data_integration/spark_init.py"
PRODUCER_SCRIPT="src/data_integration/producer.py"
CONSUMER_SCRIPT="src/data_integration/consumer.py"
BRONZE_TO_SILVER_SCRIPT="src/data_processing/bronze_to_silver_processing.py"
SILVER_TO_GOLD_SCRIPT="src/data_processing/silver_to_gold_processing.py"

# Function to check if a process completed successfully
check_status() {
    if [ $? -eq 0 ]; then
        echo "$1 completed successfully"
    else
        echo "$1 failed"
        exit 1
    fi
}

# Function to wait for a specific time and check if a process is still running
wait_for_completion() {
    local pid=$1
    local process_name=$2
    local timeout=300  # 5 minutes timeout
    local counter=0

    while kill -0 $pid 2>/dev/null; do
        sleep 10
        counter=$((counter + 10))

        if [ $counter -ge $timeout ]; then
            echo "$process_name is taking too long. Killing process..."
            kill -9 $pid
            exit 1
        fi
    done

    wait $pid
    check_status "$process_name"
}

echo "Starting data pipeline..."

# Initialize Spark session
echo "Initializing Spark session..."
python3.11 $SPARK_INIT_SCRIPT &
SPARK_PID=$!
wait_for_completion $SPARK_PID "Spark initialization"

# Start the producer
echo "Starting producer..."
python3.11 $PRODUCER_SCRIPT &
PRODUCER_PID=$!
wait_for_completion $PRODUCER_PID "Producer"

# Start the consumer
echo "Starting consumer..."
python3.11 $CONSUMER_SCRIPT &
CONSUMER_PID=$!
wait_for_completion $CONSUMER_PID "Consumer"

# Start bronze to silver transformation
echo "Starting bronze to silver transformation..."
python3.11 $BRONZE_TO_SILVER_SCRIPT &
BRONZE_TO_SILVER_PID=$!
wait_for_completion $BRONZE_TO_SILVER_PID "Bronze to Silver transformation"

# Start silver to gold transformation
echo "Starting silver to gold transformation..."
python3.11 $SILVER_TO_GOLD_SCRIPT &
SILVER_TO_GOLD_PID=$!
wait_for_completion $SILVER_TO_GOLD_PID "Silver to Gold transformation"

echo "Pipeline execution completed successfully!"