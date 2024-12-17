#!/bin/bash

command_exists() {
    command -v "$1" &> /dev/null
}

# Check if Python 3.11 is installed
if ! command_exists python3.11; then
    echo "Python 3.11 is required but not installed. Run the python.sh script to install python 3.11"
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
python3.11 -m ensurepip --upgrade

# Upgrade pip
python3.11 -m pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p src/data/{bronze,silver,gold}

# Run the pipeline
bash scripts/pipeline.sh

# Deactivate the virtual environment after completion
deactivate