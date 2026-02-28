#!/usr/bin/env bash
# =====================================================
# Enable Log Environment Variables
# =====================================================

# 使用第一个参数作为日志级别，默认 0
LOG_LEVEL=${1:-0}

# 简单校验（可选）
if ! [[ "$LOG_LEVEL" =~ ^[0-9]+$ ]]; then
    echo "Error: LOG_LEVEL must be a number"
    exit 1
fi

echo "==> Enabling log related environment variables..."
echo "==> Using LOG_LEVEL=${LOG_LEVEL}"

# Ascend global log level
export ASCEND_GLOBAL_LOG_LEVEL=${LOG_LEVEL}

# Ascend device log level
export ASCEND_DEVICE_LOG_LEVEL=${LOG_LEVEL}

# Framework global log level
export GLOBAL_LOG_LEVEL=${LOG_LEVEL}

# Log path
export ASCEND_PROCESS_LOG_PATH="$(pwd)"

echo "==> Log environment variables enabled:"
echo "    ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL}"
echo "    ASCEND_DEVICE_LOG_LEVEL=${ASCEND_DEVICE_LOG_LEVEL}"
echo "    GLOBAL_LOG_LEVEL=${GLOBAL_LOG_LEVEL}"
echo "    ASCEND_PROCESS_LOG_PATH=${ASCEND_PROCESS_LOG_PATH}"
