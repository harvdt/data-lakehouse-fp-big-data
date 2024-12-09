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
LOG_LEVEL="info"
MAX_WORKERS=4
API_HOST="0.0.0.0"
REQUIREMENTS_FILE="$API_DIR/requirements.txt"
API_SCRIPT="$API_DIR/main.py"

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
    mkdir -p "$TRAINING_DIR/trained_models" "$API_DIR" "$LOG_DIR"
}

# Create requirements.txt
create_requirements() {
    echo "Creating requirements.txt..."
    cat > "$REQUIREMENTS_FILE" << EOL
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
xgboost==2.0.2
delta-spark==3.0.0
pyspark==3.5.0
python-multipart==0.0.6
pydantic==2.5.2
EOL
}

# Install requirements
install_requirements() {
    echo "Installing required Python packages..."
    pip install -r "$REQUIREMENTS_FILE" || handle_error "Package installation failed"
    echo "Packages installed successfully"
}

# Copy API script
copy_api_script() {
    echo "Setting up API script..."
    cp "$SRC_DIR/models/api/main.py" "$API_SCRIPT" || handle_error "Failed to copy API script"
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
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "Model training completed successfully"
    else
        handle_error "Model training failed. Check $LOG_DIR/training.log for details"
    fi
}


# Start API
start_api() {
    echo "Starting FastAPI server..."
    cd "$API_DIR" || handle_error "Failed to change to API directory"
    
    # Kill any existing process on the same port
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null
    
    # Start the API server
    uvicorn main:app --host $API_HOST --port $API_PORT \
        --workers $MAX_WORKERS --log-level $LOG_LEVEL \
        2>&1 | tee "$LOG_DIR/api.log" &
    
    API_PID=$!
    
    # Wait for API to start
    echo "Waiting for API server to start..."
    sleep 5
    
    # Check if API is running
    if kill -0 $API_PID 2>/dev/null; then
        echo "API server started successfully on http://$API_HOST:$API_PORT"
        echo "API Documentation available at http://$API_HOST:$API_PORT/docs"
    else
        handle_error "Failed to start API server"
    fi
}

# Health check function
check_api_health() {
    echo "Checking API health..."
    for i in {1..30}; do
        if curl -s "http://$API_HOST:$API_PORT/health" > /dev/null; then
            echo "API is healthy!"
            return 0
        fi
        sleep 1
    done
    handle_error "API health check failed"
}

# Main execution
main() {
    echo "Starting ML Pipeline Setup and API Deployment..."
    echo "Project root: $PROJECT_ROOT"
    
    echo "Activating virtual environment..."
    source venv/bin/activate

    # Check if gold data exists
    check_gold_data
    
    # Setup environment
    setup_directories
    create_requirements
    install_requirements
    
    # Copy API script if it doesn't exist
    if [ ! -f "$API_SCRIPT" ]; then
        copy_api_script
    fi
    
    # Train or update model if needed
    if [ ! -f "$TRAINING_DIR/trained_models/sales_predictor.joblib" ] || [ "$1" == "--force-train" ]; then
        echo "Training new model..."
        train_model
    else
        echo "Using existing trained model"
    fi
    
    # Start API server
    start_api
    
    # Perform health check
    check_api_health
    
    echo "ML Pipeline and API are running."
    echo "API logs: $LOG_DIR/api.log"
    echo "Training logs: $LOG_DIR/training.log"
    echo "API Documentation: http://$API_HOST:$API_PORT/docs"
    echo "Press Ctrl+C to stop."
    
    # Setup cleanup trap
    trap 'kill $API_PID 2>/dev/null' EXIT INT TERM
    
    # Keep script running and monitor API process
    while kill -0 $API_PID 2>/dev/null; do
        sleep 10
    done
}

# Run main function with any passed arguments
main "$@"