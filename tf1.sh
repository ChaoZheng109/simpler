#!/bin/bash

# Default values
TOTAL=100
PASS=0
FAIL=0
EXAMPLE=""
DEVICE="10"
PLATFORM="a2a3"

# Parse command line arguments
while getopts "e:d:p:" opt; do
    case $opt in
        e) EXAMPLE="$OPTARG" ;;
        d) DEVICE="$OPTARG" ;;
        p) PLATFORM="$OPTARG" ;;
        *) echo "Usage: $0 -e <bg|pa|bpa|ve> [-d device] [-p platform]"
           exit 1 ;;
    esac
done

# Validate and map example name
case "$EXAMPLE" in
    bg)
        EXAMPLE_NAME="bgemm"
        EXAMPLE_DIR="bgemm"
        ;;
    pa)
        EXAMPLE_NAME="paged_attention"
        EXAMPLE_DIR="paged_attention"
        ;;
    bpa)
        EXAMPLE_NAME="batch_paged_attention"
        EXAMPLE_DIR="batch_paged_attention"
        ;;
    ve)
        EXAMPLE_NAME="vector_example"
        EXAMPLE_DIR="vector_example"
        ;;
    *)
        echo "Error: Invalid example. Use -e <bg|pa|bpa|ve>"
        echo "  bg: bgemm"
        echo "  pa: paged_attention"
        echo "  bpa: batch_paged_attention"
        echo "  ve: vector_example"
        exit 1
        ;;
esac

LOGFILE="batch_run_$(date '+%Y%m%d_%H%M%S')-${EXAMPLE_NAME}_${PLATFORM}.txt"

CMD="python examples/scripts/run_example.py \
    -k examples/tensormap_and_ringbuffer/${EXAMPLE_DIR}/kernels \
    -g examples/tensormap_and_ringbuffer/${EXAMPLE_DIR}/golden.py \
    -p ${PLATFORM} -d=${DEVICE}"

echo "Log file: $LOGFILE"
echo "Running example: $EXAMPLE_NAME (platform: $PLATFORM, device: $DEVICE)"
echo "Batch run started at $(date)" | tee "$LOGFILE"
echo "" | tee -a "$LOGFILE"

for i in $(seq 1 $TOTAL); do
    echo "=== Run $i / $TOTAL ===" | tee -a "$LOGFILE"
    output=$($CMD 2>&1)
    echo "$output" >> "$LOGFILE"
    if echo "$output" | grep -q "TEST PASSED"; then
        PASS=$((PASS + 1))
        echo "[Run $i] PASSED" | tee -a "$LOGFILE"
    else
        FAIL=$((FAIL + 1))
        echo "[Run $i] FAILED - Stopping execution" | tee -a "$LOGFILE"
        echo "" >> "$LOGFILE"
        break
    fi
    echo "" >> "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "==============================" | tee -a "$LOGFILE"
echo "Results: $PASS / $TOTAL PASSED" | tee -a "$LOGFILE"
echo "Batch run finished at $(date)" | tee -a "$LOGFILE"
echo "==============================" | tee -a "$LOGFILE"
echo "Full log saved to: $LOGFILE"