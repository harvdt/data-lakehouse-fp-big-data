#!/bin/bash

# Set path variables
PROJECT_ROOT="$(dirname "$(pwd)")"  # Go up one level from scripts
SRC_DIR="$PROJECT_ROOT/src"
MODEL_DIR="$SRC_DIR/models"
TRAINING_DIR="$MODEL_DIR/training"
API_DIR="$MODEL_DIR/api"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$SRC_DIR/data"
GOLD_DIR="$DATA_DIR/gold"

# Create all necessary directories
echo "Creating directory structure..."
mkdir -p "$TRAINING_DIR/trained_models"
mkdir -p "$API_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$GOLD_DIR"

# Check if gold data exists
check_gold_data() {
    if [ ! -d "$GOLD_DIR/revenue_per_category" ]; then
        echo "Error: Gold layer data not found at $GOLD_DIR"
        echo "Please run the silver to gold transformation first."
        exit 1
    fi
}

# Configuration
API_PORT=8000
LOG_LEVEL="info"  # Changed to lowercase
MAX_WORKERS=4

# Error handling
handle_error() {
    local error_msg="$1"
    local error_code="${2:-1}"
    echo "Error: $error_msg" | tee -a "$LOG_DIR/error.log"
    exit "$error_code"
}

# Setup directories
setup_directories() {
    echo "Setting up directory structure..."
    mkdir -p "$TRAINING_DIR/trained_models" "$API_DIR" "$LOG_DIR" "$BACKUP_PATH"
}

# Install requirements
install_requirements() {
    echo "Installing required Python packages..."
    pip install fastapi==0.104.1 \
                uvicorn==0.24.0 \
                pandas==2.1.3 \
                numpy==1.26.2 \
                scikit-learn==1.3.2 \
                joblib==1.3.2 \
                xgboost==2.0.2 \
                delta-spark==3.0.0 \
                pyspark==3.5.0 \
                matplotlib==3.8.2 \
                seaborn==0.13.0 \
                requests==2.31.0 || handle_error "Package installation failed"
    echo "Packages installed successfully"
}

# Train model
train_model() {
    echo "Starting model training..."
    cd "$TRAINING_DIR" || handle_error "Failed to change to training directory"
    
    if [ ! -f "model_training.py" ]; then
        echo "Model training script not found. Creating directories and copying files..."
        mkdir -p "$TRAINING_DIR"
        cp "$SRC_DIR/data_processing/bronze_to_silver/model_training.py" "$TRAINING_DIR/" || handle_error "Failed to copy model training script"
    fi
    
    python model_training.py 2>&1 | tee "$LOG_DIR/training.log"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Model training completed successfully"
    else
        handle_error "Model training failed. Check $LOG_DIR/training.log for details"
    fi
}

# Start API
start_api() {
    echo "Starting FastAPI server..."
    cd "$API_DIR" || handle_error "Failed to change to API directory"
    
    if [ ! -f "main.py" ]; then
        echo "API script not found. Creating directories and copying files..."
        mkdir -p "$API_DIR"
        cp "$SRC_DIR/data_processing/bronze_to_silver/main.py" "$API_DIR/" || handle_error "Failed to copy API script"
    fi
    
    uvicorn main:app --reload --host 0.0.0.0 --port "$API_PORT" \
        --workers "$MAX_WORKERS" --log-level "$LOG_LEVEL" \
        2>&1 | tee "$LOG_DIR/api.log" &
    API_PID=$!
    
    echo "Waiting for API server to start..."
    sleep 5
}

# Main execution
main() {
    echo "Starting ML Pipeline Setup and Execution..."
    echo "Project root: $PROJECT_ROOT"
    
    setup_directories
    install_requirements
    
    if [ ! -f "$TRAINING_DIR/trained_models/sales_predictor.joblib" ] || [ "$1" == "--force-train" ]; then
        echo "Training new model..."
        train_model
    else
        echo "Using existing trained model"
    fi
    
    start_api
    
    echo "ML Pipeline is running."
    echo "API logs: $LOG_DIR/api.log"
    echo "Training logs: $LOG_DIR/training.log"
    echo "Press Ctrl+C to stop."
    
    # Setup cleanup trap
    trap 'kill $API_PID 2>/dev/null' EXIT INT TERM
    
    # Keep script running
    wait $API_PID
}

# Run main function
main "$@"