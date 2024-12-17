#!/bin/bash

# Function to download and copy a dataset
download_and_copy() {
    local file_id=$1
    local output_name=$2
    wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${file_id}" -O "${output_name}"
    cp "${output_name}" ./src/data/raw
}

# Ensure argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/simulation.sh [3|4|all]"
    exit 1
fi

# Process based on the argument
if [ "$1" -eq 1 ]; then
    echo "Downloading dataset 3"
    download_and_copy "1zyEiOLaK5q88v6Rht3CyLWRruDl0T5e0" "ecommerce_product_dataset3.csv"
elif [ "$1" -eq 2 ]; then
    echo "Downloading dataset 4"
    download_and_copy "1PBSOP679FACYkeN6Zs9TACbLYUcby-Ql" "ecommerce_product_dataset4.csv"
elif [ "$1" == "all" ]; then
    echo "Downloading all datasets..."
    download_and_copy "1zyEiOLaK5q88v6Rht3CyLWRruDl0T5e0" "ecommerce_product_dataset3.csv"
    download_and_copy "1PBSOP679FACYkeN6Zs9TACbLYUcby-Ql" "ecommerce_product_dataset4.csv"
else
    echo "Invalid argument. Use 3 for dataset 3 or 4 for dataset 4."
    exit 1
fi

echo "Dataset downloaded successfully!"
