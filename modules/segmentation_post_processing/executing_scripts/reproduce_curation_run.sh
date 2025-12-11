#!/bin/bash

# Script to reproduce a previous segmentation curation run
# Usage: ./reproduce_run.sh <run_id>
# Example: ./reproduce_run.sh curation_20241216_143022

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <run_id>"
    echo ""
    echo "Available runs:"
    ls -1 logs/*_parameters.txt 2>/dev/null | sed 's/logs\///; s/_parameters.txt//' | sort -r | head -10
    exit 1
fi

RUN_ID=$1
PARAMS_FILE="logs/${RUN_ID}_parameters.txt"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: Parameters file not found: $PARAMS_FILE"
    echo ""
    echo "Available runs:"
    ls -1 logs/*_parameters.txt 2>/dev/null | sed 's/logs\///; s/_parameters.txt//' | sort -r | head -10
    exit 1
fi

echo "Reproducing run: $RUN_ID"
echo "Reading parameters from: $PARAMS_FILE"
echo ""

# Extract and execute the command from the parameters file
CMD=$(grep -A 1 "Command Line:" "$PARAMS_FILE" | tail -1)

if [ -z "$CMD" ]; then
    echo "Error: Could not extract command from parameters file"
    exit 1
fi

echo "Executing: $CMD"
echo ""

# Execute the command
eval "$CMD"