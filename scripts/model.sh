#!/bin/bash

# Set path variables
PROJECT_ROOT="$(dirname "$(pwd)")"  # Go up one level from scripts
SRC_DIR="$PROJECT_ROOT/data-lakehouse-fp-big-data/src"
MODEL_DIR="$SRC_DIR/models"
TRAINING_DIR="$MODEL_DIR/training"
API_DIR="$MODEL_DIR/api"
LOG_DIR="$PROJECT_ROOT/logs"
DATA_DIR="$SRC_DIR/data"
GOLD_DIR="$DATA_DIR/gold"

# Configuration
API_PORT=8000
LOG_LEVEL="info"
MAX_WORKERS=4
API_HOST="0.0.0.0"
REQUIREMENTS_FILE="$API_DIR/requirements.txt"
API_SCRIPT="$API_DIR/main.py"
TRAINED_MODEL="$TRAINING_DIR/trained_models/sales_predictor.joblib"

# Create all necessary directories
mkdir -p "$TRAINING_DIR/trained_models" "$API_DIR" "$LOG_DIR"

# Error handling
handle_error() {
    echo "Error: $1" | tee -a "$LOG_DIR/error.log"
    exit "${2:-1}"
}

# Train model
train_model() {
    echo "Starting model training..."
    cd "$TRAINING_DIR" || handle_error "Failed to change to training directory"

    # Ensure training script exists
    if [ ! -f "model_training.py" ]; then
        cp "$SRC_DIR/data_processing/model_training.py" . || handle_error "Failed to copy model training script"
    fi

    # Run the training script
    python model_training.py 2>&1 | tee "$LOG_DIR/training.log"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        handle_error "Model training failed. Check $LOG_DIR/training.log for details"
    fi

    echo "Model training completed successfully"
}

# Main execution
main() {
    # Ensure API script exists
    if [ ! -f "$API_SCRIPT" ]; then
        cp "$SRC_DIR/models/api/main.py" "$API_SCRIPT" || handle_error "Failed to copy API script"
    fi

    # Train model if it doesn't exist or if forced
    if [ ! -f "$TRAINED_MODEL" ] || [ "$1" == "--force-train" ]; then
        train_model
    else
        echo "Using existing trained model"
    fi

    # Start web services
    bash scripts/web.sh || handle_error "Failed to start web services"

	wait
}

main "$@"