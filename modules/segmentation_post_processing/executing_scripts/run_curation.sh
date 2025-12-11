#!/bin/bash

# Multithreaded Segmentation Curation Pipeline Runner (Robust Version 3)
# This script is designed to be run from any directory. It will automatically
# locate the project root, activate the venv, and execute everything from there
# using multithreaded processing for improved performance.

set -e  # Exit on any error
set -o pipefail # Ensures that the exit code of a pipeline is the rightmost command to exit with a non-zero status

# --- DYNAMIC PATH SETUP ---
# Find the directory where this script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Navigate up to the project root (assuming the script is in executing_scripts/segmentation_curation).
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/../" &> /dev/null && pwd )

# Change the current directory to the project root.
cd "$PROJECT_ROOT"
echo "✅ Successfully set working directory to project root: $PROJECT_ROOT"

# --- RELATIVE SCRIPT PATHS ---
PYTHON_SCRIPT_DIR="post_raw_segmentation_curration"
PYTHON_RUNNER="$PYTHON_SCRIPT_DIR/run_segmentation_curation.py"
PYTHON_CURATOR="$PYTHON_SCRIPT_DIR/segmentation_curator.py"

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR="/Users/noahbruderer/Desktop/250707_R3_timeseries_inhibitors_CF_seg"
MIN_CELL_SIZE=50
MAX_SMALL_CELL=50
OUTPUT_SUFFIX="_cur"
PATTERN="_seg"
SAVE_VISUALIZATIONS=true
VERBOSE=true
THREADS=12  # Number of worker threads for parallel processing

# =============================================================================
# SETUP AND LOGGING
# =============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="curation_mt_${TIMESTAMP}"  # Added 'mt' for multithreaded
LOGS_DIR="./logs"
mkdir -p "$LOGS_DIR"
LOG_FILE="${LOGS_DIR}/${RUN_ID}.log"
PARAMS_FILE="${LOGS_DIR}/${RUN_ID}_parameters.txt"
SUMMARY_FILE="${LOGS_DIR}/${RUN_ID}_summary.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# --- EXPLICIT VENV ACTIVATION ---
log "Locating and activating the Poetry virtual environment..."
VENV_PATH=$(poetry env info --path)
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    log "ERROR: Could not find the virtual environment's activate script at $VENV_PATH/bin/activate"
    log "Try running 'poetry install' from the project root."
    exit 1
fi
source "$VENV_PATH/bin/activate"
log "✅ Virtual environment activated successfully."
# --- END VENV ACTIVATION ---

log_parameters() {
    # Reconstruct the command that will be run (NO 'poetry run' needed now)
    CMD="python3 \"$PYTHON_RUNNER\" \"$INPUT_DIR\""
    CMD="$CMD --min-cell-size $MIN_CELL_SIZE"
    CMD="$CMD --max-small-cell $MAX_SMALL_CELL"
    CMD="$CMD --suffix \"$OUTPUT_SUFFIX\""
    CMD="$CMD --pattern \"$PATTERN\""
    CMD="$CMD --threads $THREADS"
    
    if [ "$SAVE_VISUALIZATIONS" = true ]; then
        CMD="$CMD --visualizations"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi

    cat > "$PARAMS_FILE" << EOF
Multithreaded Segmentation Curation Run Parameters
===================================================
Run ID: $RUN_ID
Timestamp: $(date '+%Y-%m-%d %H:%M:%S')
Host: $(hostname)
User: $(whoami)
Working Directory: $(pwd)

Input Parameters:
-----------------
INPUT_DIR: $INPUT_DIR
MIN_CELL_SIZE: $MIN_CELL_SIZE
MAX_SMALL_CELL: $MAX_SMALL_CELL
OUTPUT_SUFFIX: $OUTPUT_SUFFIX
PATTERN: $PATTERN
SAVE_VISUALIZATIONS: $SAVE_VISUALIZATIONS
VERBOSE: $VERBOSE
THREADS: $THREADS

System Information:
-------------------
CPU Cores Available: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")
Memory (GB): $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}' || echo "Unknown")
Platform: $(uname -s)
Architecture: $(uname -m)

Python Environment:
-------------------
Python Version: $(python3 --version 2>&1)
Python Path: $(which python3)
Virtual Environment: $VENV_PATH

Threading Configuration:
------------------------
Worker Threads: $THREADS
Expected Performance: ~${THREADS}x speedup (theoretical maximum)

Command Line:
-------------
$CMD
EOF
}

# =============================================================================
# SYSTEM CHECKS AND OPTIMIZATION
# =============================================================================
log "Performing system checks for multithreaded processing..."

# Check CPU cores
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "0")
if [ "$CPU_CORES" -gt 0 ]; then
    log "Detected $CPU_CORES CPU cores"
    if [ "$THREADS" -gt "$CPU_CORES" ]; then
        log "WARNING: Using $THREADS threads on $CPU_CORES cores may cause oversubscription"
        log "Consider reducing --threads to $CPU_CORES or less for optimal performance"
    else
        log "Thread configuration ($THREADS threads) is appropriate for this system"
    fi
else
    log "WARNING: Could not detect CPU core count, proceeding with $THREADS threads"
fi

# Check if system supports the threading approach
log "Checking threading support..."
python3 -c "
import concurrent.futures
import threading
print(f'Threading module available: {threading.active_count() >= 0}')
print(f'ThreadPoolExecutor available: {hasattr(concurrent.futures, \"ThreadPoolExecutor\")}')
" 2>/dev/null || {
    log "ERROR: Threading modules not available. Check Python installation."
    exit 1
}

# =============================================================================
# VALIDATION
# =============================================================================
log "Starting multithreaded segmentation curation pipeline"
log "Run ID: $RUN_ID"

if [ ! -d "$INPUT_DIR" ]; then log "ERROR: Input directory does not exist: $INPUT_DIR"; exit 1; fi
if [ ! -f "$PYTHON_RUNNER" ]; then log "ERROR: run_segmentation_curation.py not found at: $PYTHON_RUNNER"; exit 1; fi
if [ ! -f "$PYTHON_CURATOR" ]; then log "ERROR: segmentation_curator.py not found at: $PYTHON_CURATOR"; exit 1; fi

# Check if the script supports multithreading
if ! grep -q "ThreadPoolExecutor" "$PYTHON_RUNNER"; then
    log "ERROR: The Python script does not appear to support multithreading"
    log "Make sure you're using the multithreaded version of run_segmentation_curation.py"
    exit 1
fi

log "Checking Python dependencies..."
# NO 'poetry run' needed as the venv is now active for this script session
python3 -c "import numpy, plotly, tqdm, pathlib, concurrent.futures" 2>/dev/null || {
    log "ERROR: Missing required Python packages. Your venv might be corrupted. Try 'poetry install'."
    exit 1
}
log "All dependencies satisfied"

# =============================================================================
# PERFORMANCE ESTIMATION
# =============================================================================
log "Estimating file count for performance prediction..."
FILE_COUNT=$(find "$INPUT_DIR" -name "*${PATTERN}*.tif" -o -name "*${PATTERN}*.tiff" 2>/dev/null | wc -l | tr -d ' ')
if [ "$FILE_COUNT" -gt 0 ]; then
    log "Found approximately $FILE_COUNT files to process"
    
    # Rough time estimation (assuming 2-10 seconds per file on average)
    MIN_TIME_PER_FILE=2
    MAX_TIME_PER_FILE=10
    
    SEQUENTIAL_MIN=$((FILE_COUNT * MIN_TIME_PER_FILE))
    SEQUENTIAL_MAX=$((FILE_COUNT * MAX_TIME_PER_FILE))
    
    PARALLEL_MIN=$((SEQUENTIAL_MIN / THREADS))
    PARALLEL_MAX=$((SEQUENTIAL_MAX / THREADS))
    
    log "Performance estimate:"
    log "  Sequential processing: ${SEQUENTIAL_MIN}-${SEQUENTIAL_MAX} seconds ($(printf "%.1f" $(echo "$SEQUENTIAL_MIN/60" | bc -l))-$(printf "%.1f" $(echo "$SEQUENTIAL_MAX/60" | bc -l)) minutes)"
    log "  Parallel processing ($THREADS threads): ${PARALLEL_MIN}-${PARALLEL_MAX} seconds ($(printf "%.1f" $(echo "$PARALLEL_MIN/60" | bc -l))-$(printf "%.1f" $(echo "$PARALLEL_MAX/60" | bc -l)) minutes)"
    log "  Expected speedup: ~${THREADS}x"
else
    log "WARNING: No files found matching pattern '*${PATTERN}*.tif' in $INPUT_DIR"
fi

# =============================================================================
# LOG PARAMETERS & RUN CURATION
# =============================================================================
log_parameters
log "Parameters logged to: $PARAMS_FILE"

CMD=$(tail -n 1 "$PARAMS_FILE")

log "Executing multithreaded curation with $THREADS worker threads:"
log "$CMD"
log "Output will be logged to: $LOG_FILE"

START_TIME=$(date +%s)
log "Starting multithreaded curation process..."

# Run the command and capture both the exit code and timing
eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
RETURN_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================
if [ $RETURN_CODE -eq 0 ] && [ "$FILE_COUNT" -gt 0 ] && [ "$DURATION" -gt 0 ]; then
    THROUGHPUT=$(echo "scale=2; $FILE_COUNT / $DURATION" | bc -l)
    TIME_PER_FILE=$(echo "scale=2; $DURATION / $FILE_COUNT" | bc -l)
    
    log "Performance metrics:"
    log "  Files processed: $FILE_COUNT"
    log "  Total time: $DURATION seconds"
    log "  Throughput: $THROUGHPUT files/second"
    log "  Average time per file: $TIME_PER_FILE seconds"
    
    # Estimate sequential time for comparison
    ESTIMATED_SEQUENTIAL=$((FILE_COUNT * 5))  # Assume 5 seconds per file sequentially
    if [ "$ESTIMATED_SEQUENTIAL" -gt "$DURATION" ]; then
        ACTUAL_SPEEDUP=$(echo "scale=1; $ESTIMATED_SEQUENTIAL / $DURATION" | bc -l)
        log "  Estimated speedup vs sequential: ${ACTUAL_SPEEDUP}x"
    fi
fi

# =============================================================================
# SUMMARY & COMPLETION
# =============================================================================
log "Multithreaded curation process completed with return code: $RETURN_CODE"
log "Total duration: $DURATION seconds"

cat > "$SUMMARY_FILE" << EOF
Multithreaded Segmentation Curation Run Summary
================================================
Run ID: $RUN_ID
Start Time: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $START_TIME '+%Y-%m-%d %H:%M:%S')
End Time: $(date -d @$END_TIME '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r $END_TIME '+%Y-%m-%d %H:%M:%S')
Duration: $DURATION seconds
Return Code: $RETURN_CODE
Status: $([ $RETURN_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")

Parameters Used:
----------------
INPUT_DIR: $INPUT_DIR
MIN_CELL_SIZE: $MIN_CELL_SIZE
MAX_SMALL_CELL: $MAX_SMALL_CELL
OUTPUT_SUFFIX: $OUTPUT_SUFFIX
PATTERN: $PATTERN
SAVE_VISUALIZATIONS: $SAVE_VISUALIZATIONS
THREADS: $THREADS

Performance:
------------
Files Processed: $FILE_COUNT
$([ "$FILE_COUNT" -gt 0 ] && [ "$DURATION" -gt 0 ] && echo "Throughput: $(echo "scale=2; $FILE_COUNT / $DURATION" | bc -l) files/second" || echo "Throughput: N/A")
$([ "$FILE_COUNT" -gt 0 ] && [ "$DURATION" -gt 0 ] && echo "Avg Time/File: $(echo "scale=2; $DURATION / $FILE_COUNT" | bc -l) seconds" || echo "Avg Time/File: N/A")

System Info:
------------
CPU Cores: $CPU_CORES
Host: $(hostname)
Platform: $(uname -s) $(uname -m)

Files:
------
Log File: $LOG_FILE
Parameters File: $PARAMS_FILE
Summary File: $SUMMARY_FILE

To reproduce this run:
----------------------
$CMD
EOF

log "Summary written to: $SUMMARY_FILE"

if [ $RETURN_CODE -eq 0 ]; then
    log "✅ Multithreaded segmentation curation completed successfully!"
    log "🚀 Used $THREADS worker threads for parallel processing"
else
    log "❌ Multithreaded segmentation curation failed with return code: $RETURN_CODE"
    log "📋 Check the log file for details: $LOG_FILE"
fi

echo
echo "Run ID: $RUN_ID"
echo "Status: $([ $RETURN_CODE -eq 0 ] && echo "SUCCESS ✅" || echo "FAILED ❌")"
echo "Duration: $DURATION seconds"
echo "Threads: $THREADS"
echo "Log: $LOG_FILE"