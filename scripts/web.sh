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

# Function to start the frontend
start_frontend() {
    echo "Starting frontend..."

    cd frontend

    pnpm i

    pnpm run dev
}

mkdir -p "$LOG_DIR"

start_frontend
