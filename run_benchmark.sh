#!/bin/bash

# Benchmark script for paged_attention test
# Runs N iterations and collects timing statistics
# Usage: ./run_benchmark.sh [num_iterations]
#   num_iterations: Number of times to run the test (default: 10)

# Configuration
NUM_ITERATIONS=${1:-10}
OUTPUT_DIR="benchmark_results"
LOG_DIR="log/debug/device-13"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${OUTPUT_DIR}/benchmark_${TIMESTAMP}.txt"
RUN_OUTPUT_FILE="${OUTPUT_DIR}/run_output_${TIMESTAMP}.txt"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Clear the run output file
> "${RUN_OUTPUT_FILE}"

# Array to store execution times
declare -a exec_times

echo "Starting benchmark: ${NUM_ITERATIONS} iterations of paged_attention test"
echo "Run output will be saved to: ${RUN_OUTPUT_FILE}"
echo "Results will be saved to: ${RESULTS_FILE}"
echo ""

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    echo "Running iteration $i/${NUM_ITERATIONS}..."

    # Get the latest log file before running
    BEFORE_LOG=$(ls -t "${LOG_DIR}"/device-*.log 2>/dev/null | head -n 1)

    # Run the test and append output to the same file
    echo "========================================" >> "${RUN_OUTPUT_FILE}"
    echo "Run $i/${NUM_ITERATIONS} - $(date)" >> "${RUN_OUTPUT_FILE}"
    echo "========================================" >> "${RUN_OUTPUT_FILE}"
    python examples/scripts/run_example.py \
        -k tests/device_tests/host_build_graph/paged_attention/kernels \
        -g tests/device_tests/host_build_graph/paged_attention/golden.py \
        -p a2a3 \
        -d 13 \
        >> "${RUN_OUTPUT_FILE}" 2>&1
    echo "" >> "${RUN_OUTPUT_FILE}"

    # Wait a moment for log file to be written
    sleep 1

    # Get the latest log file after running
    AFTER_LOG=$(ls -t "${LOG_DIR}"/device-*.log 2>/dev/null | head -n 1)

    # Check if a new log file was created
    if [ "$BEFORE_LOG" != "$AFTER_LOG" ] && [ -f "$AFTER_LOG" ]; then
        echo "  Log file: $AFTER_LOG"

        # Extract timestamps from log messages
        # Looking for lines like: "Thread X: Main execution loop starting at 2621578630939.380us"
        # and "Thread X: Main execution loop ended at 2621578644345.140us"

        # Extract all start times (keep decimal precision)
        MIN_START=$(grep "starting at" "$AFTER_LOG" | \
                    sed -n 's/.*starting at \([0-9.]*\)us.*/\1/p' | \
                    sort -n | head -n 1)

        # Extract all end times (keep decimal precision)
        MAX_END=$(grep "ended at" "$AFTER_LOG" | \
                  sed -n 's/.*ended at \([0-9.]*\)us.*/\1/p' | \
                  sort -n | tail -n 1)

        if [ -n "$MIN_START" ] && [ -n "$MAX_END" ]; then
            # Calculate execution time (in microseconds) with 2 decimal places
            EXEC_TIME=$(echo "$MAX_END - $MIN_START" | bc | awk '{printf "%.2f", $1}')
            exec_times+=($EXEC_TIME)
            echo "  Execution time: $EXEC_TIME us"
        else
            echo "  Warning: Could not extract timing from log file"
            exec_times+=(0)
        fi
    else
        echo "  Warning: No new log file found"
        exec_times+=(0)
    fi

    echo ""
done

# Calculate statistics
echo "========================================"
echo "Benchmark Results"
echo "========================================"

# Filter out zero values (failed runs)
valid_times=()
for time in "${exec_times[@]}"; do
    # Check if time is greater than 0 (using bc for float comparison)
    if [ "$(echo "$time > 0" | bc)" -eq 1 ]; then
        valid_times+=($time)
    fi
done

if [ ${#valid_times[@]} -eq 0 ]; then
    echo "Error: No valid timing data collected"
    exit 1
fi

# Find min, max, and calculate average
min=${valid_times[0]}
max=${valid_times[0]}
sum=0

for time in "${valid_times[@]}"; do
    # Compare using bc for floating point
    if [ "$(echo "$time < $min" | bc)" -eq 1 ]; then
        min=$time
    fi
    if [ "$(echo "$time > $max" | bc)" -eq 1 ]; then
        max=$time
    fi
    sum=$(echo "$sum + $time" | bc)
done

# Calculate average with 2 decimal places
avg=$(echo "scale=2; $sum / ${#valid_times[@]}" | bc)

echo "Valid runs: ${#valid_times[@]}/${NUM_ITERATIONS}"
echo "Fastest execution: $min us"
echo "Slowest execution: $max us"
echo "Average execution: $avg us"
echo ""
echo "All execution times:"
for i in "${!exec_times[@]}"; do
    echo "  Run $((i+1)): ${exec_times[$i]} us"
done

# Save results to file
{
    echo "Benchmark Results - $(date)"
    echo "========================================"
    echo "Test: paged_attention"
    echo "Device: 13"
    echo "Valid runs: ${#valid_times[@]}/${NUM_ITERATIONS}"
    echo ""
    echo "Fastest execution: $min us"
    echo "Slowest execution: $max us"
    echo "Average execution: $avg us"
    echo ""
    echo "Individual run times:"
    for i in "${!exec_times[@]}"; do
        echo "  Run $((i+1)): ${exec_times[$i]} us"
    done
} > "${RESULTS_FILE}"

echo "Results saved to: ${RESULTS_FILE}"
echo "All run outputs saved to: ${RUN_OUTPUT_FILE}"
