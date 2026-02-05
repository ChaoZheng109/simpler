/**
 * @file device_log.cpp
 * @brief Simulation Platform Log Implementation
 *
 * Provides log enable flags and initialization for simulation environment.
 * Simulation always enables all log levels by default.
 */

#include "aicpu/device_log.h"
#include <cstdarg>
#include <cstdio>

// =============================================================================
// Log Enable Flags (Simulation: always enabled)
// =============================================================================

bool g_is_log_enable_debug = true;
bool g_is_log_enable_info = true;
bool g_is_log_enable_warn = true;
bool g_is_log_enable_error = true;

// =============================================================================
// Platform Constant
// =============================================================================

const char* TILE_FWK_DEVICE_MACHINE = "SIM_CPU";

// =============================================================================
// Log Initialization (No-op for simulation)
// =============================================================================

void init_log_switch() {
    // Simulation: no initialization needed
    // All log levels are enabled by default
    // Future: could read from environment variables if needed
}

// =============================================================================
// Platform-Specific Logging Functions (Simulation: use printf)
// =============================================================================

void dev_log_debug(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[DEBUG] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void dev_log_info(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[INFO] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void dev_log_warn(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[WARN] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

void dev_log_error(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[ERROR] %s: ", func);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}
