#!/bin/bash

# Set path variables
PROJECT_ROOT="$(dirname "$(pwd)")"  # Go up one level from scripts
SRC_DIR="$PROJECT_ROOT/data-lakehouse-fp-big-data/src"
MODEL_DIR="$SRC_DIR/models"
API_DIR="$MODEL_DIR/api"
LOG_DIR="$PROJECT_ROOT/logs"

# Configuration
API_PORT=8000
LOG_LEVEL="info"
MAX_WORKERS=4
API_HOST="0.0.0.0"
API_SCRIPT="$API_DIR/main.py"

# Create necessary directories
mkdir -p "$API_DIR" "$LOG_DIR"

# Error handling
handle_error() {
    echo "Error: $1" | tee -a "$LOG_DIR/error.log"
    exit "${2:-1}"
}

# Main execution
main() {
    # Ensure API script exists
    if [ ! -f "$API_SCRIPT" ]; then
        cp "$SRC_DIR/models/api/main.py" "$API_SCRIPT" || handle_error "Failed to copy API script"
    fi

    # Start the API server
    echo "Starting API server..."
    cd "$API_DIR" || handle_error "Failed to change to API directory"
    
    uvicorn main:app --host "$API_HOST" --port "$API_PORT" \
        --workers "$MAX_WORKERS" --log-level "$LOG_LEVEL" \
        | tee "$LOG_DIR/api.log"
}

main "$@"