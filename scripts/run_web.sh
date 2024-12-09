#!/bin/bash

# Change directory and check if successful
cd frontend || exit 1

# Install dependencies
pnpm install || exit 1

# Start development server
pnpm run dev