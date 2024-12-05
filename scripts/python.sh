#!/bin/bash

echo "Installing python3.11..."

sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 -y

python3.11 --version