#!/bin/bash

command_exists() {
    command -v "$1" &> /dev/null
}

# Check if Python 3.11 is installed
if ! command_exists python3.11; then
    echo "Python 3.11 is required but not installed. Please install Python 3.11 first."
    exit 1
fi

# Start docker services
echo "Starting Docker containers..."
docker compose up --build -d

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install pip if not present
python -m ensurepip --upgrade

# Upgrade pip
python -m pip install --upgrade pip

# Install numpy explicitly first (as it's a core dependency)
echo "Installing numpy..."
pip install numpy

# Install all requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Run the pipeline
echo "Running pipeline..."
bash scripts/pipeline.sh

# Deactivate the virtual environment after completion
deactivate