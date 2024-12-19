#!/bin/bash

# Set the project path
PROJECT_PATH="/home/zaki/kuliah/Bigdata/FP-5/data-lakehouse-fp-big-data"
cd "$PROJECT_PATH"

# Run the Python setup script first (from run.sh)
bash scripts/python.sh

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
pip install -r requirements.txt

# Run the training coordinator
python3 src/models/training/training_coordinator.py >> logs/training_coordinator.log 2>&1

# Deactivate virtual environment
deactivate