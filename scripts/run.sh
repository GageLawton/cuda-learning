#!/bin/bash

set -e

# Go to project root (so script works from anywhere)
cd "$(dirname "$0")/.."

BUILD_DIR="build"
EXECUTABLE="$BUILD_DIR/main.exe"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Build not found. Building first..."
    make
fi

echo "Running CUDA program..."
./"$EXECUTABLE"