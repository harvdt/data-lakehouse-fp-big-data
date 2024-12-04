#!/bin/bash

command_exists() {
    command -v "$1" &> /dev/null
}

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

# Check if dependencies are already installed by checking the installed packages
echo "Checking if dependencies are already installed..."
REQUIRED_PACKAGES=$(cat requirements.txt)
INSTALLED_PACKAGES=$(pip freeze)

# Compare installed packages with requirements.txt
for package in $REQUIRED_PACKAGES; do
    if echo "$INSTALLED_PACKAGES" | grep -q "$package"; then
        echo "$package is already installed."
    else
        echo "$package is not installed. Installing..."
        pip install "$package"
        if [ $? -ne 0 ]; then
            echo "Failed to install $package. Please resolve the issue manually."
            deactivate
            exit 1
        fi
    fi
done

# Run the pipeline
echo "Running pipeline..."
bash scripts/pipeline.sh

# Deactivate the virtual environment after completion
deactivate
