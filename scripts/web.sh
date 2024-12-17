#!/bin/bash

# Path variables
PROJECT_ROOT="$(dirname "$(pwd)")"
SRC_DIR="$PROJECT_ROOT/data-lakehouse-fp-big-data/src"
API_DIR="$SRC_DIR/models/api"
LOG_DIR="$PROJECT_ROOT/logs"

# Configuration variables
API_PORT=8000
LOG_LEVEL="info"
MAX_WORKERS=4
API_HOST="0.0.0.0"

# Error handling
handle_error() {
    echo "Error: $1" | tee -a "$LOG_DIR/error.log"
    exit "${2:-1}"
}

# Function to start the backend
start_backend() {
    echo "Starting backend..."
    cd "$API_DIR" || handle_error "API directory not found"

    # Kill existing process on the same port
    lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true

    # Start the API server
    uvicorn main:app --host "$API_HOST" --port "$API_PORT" \
        --workers "$MAX_WORKERS" --log-level "$LOG_LEVEL" \
        | tee "$LOG_DIR/api.log" &

    API_PID=$!

    # Wait for the API to start
    for i in {1..10}; do
        if kill -0 $API_PID 2>/dev/null; then
            echo "API server started successfully on http://$API_HOST:$API_PORT"
            echo "API Documentation available at http://$API_HOST:$API_PORT/docs"
            return 0
        fi
        sleep 1
    done

    # Handle failure to start the API
    if ! kill -0 $API_PID 2>/dev/null; then
        handle_error "Failed to start API server"
    fi
}

# Function to start the frontend
start_frontend() {
    echo "Starting frontend..."
    if [ -d "$PROJECT_ROOT/frontend" ]; then
        cd "$PROJECT_ROOT/frontend" || handle_error "Frontend directory not found"

        # Install dependencies
        pnpm i || handle_error "Failed to install frontend dependencies"

        # Start the development server
        pnpm run dev || handle_error "Failed to start frontend server"
    else
        echo "Frontend directory not found, skipping frontend setup" | tee -a "$LOG_DIR/error.log"
    fi
}

mkdir -p "$LOG_DIR"

start_backend

start_frontend
