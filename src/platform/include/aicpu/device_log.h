/**
 * @file device_log.h
 * @brief Unified Device Logging Interface for AICPU
 *
 * Provides logging macros that work on both real hardware (using CANN dlog)
 * and simulation (using printf). Uses conditional compilation to select
 * the appropriate backend with a layered design to minimize code duplication.
 *
 * Platform Support:
 * - a2a3 (__PLATFORM_A2A3__): Real hardware with CANN dlog API
 * - a2a3sim (default): Host-based simulation using printf
 */

#ifndef PLATFORM_DEVICE_LOG_H_
#define PLATFORM_DEVICE_LOG_H_

#include <cstdio>
#include <cstdint>

// =============================================================================
// Platform Detection and Thread ID
// =============================================================================

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#define GET_TID() syscall(SYS_gettid)
#else
#define GET_TID() 0
#endif

// =============================================================================
// Log Enable Flags
// =============================================================================

#if defined(__PLATFORM_A2A3__)
    // Real hardware: flags defined in device_log.cpp
    extern bool g_is_log_enable_debug;
    extern bool g_is_log_enable_info;
    extern bool g_is_log_enable_warn;
    extern bool g_is_log_enable_error;
#else
    // Simulation: always enabled
    static bool g_is_log_enable_debug = true;
    static bool g_is_log_enable_info = true;
    static bool g_is_log_enable_warn = true;
    static bool g_is_log_enable_error = true;
#endif

// =============================================================================
// Backend-Specific Logging Macros (Low-Level Layer)
// =============================================================================

#if defined(__PLATFORM_A2A3__)
    // Real hardware: use CANN dlog API
    #include "dlog_pub.h"

    constexpr const char* TILE_FWK_DEVICE_MACHINE = "AI_CPU";

    // Low-level logging backends
    #define BACKEND_LOG_DEBUG(fmt, ...) \
        dlog_debug(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__)

    #define BACKEND_LOG_INFO(fmt, ...) \
        dlog_info(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__)

    #define BACKEND_LOG_WARN(fmt, ...) \
        dlog_warn(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__)

    #define BACKEND_LOG_ERROR(fmt, ...) \
        dlog_error(AICPU, "%lu %s\n" #fmt, GET_TID(), __FUNCTION__, ##__VA_ARGS__)

#else
    // Simulation: use printf
    constexpr const char* TILE_FWK_DEVICE_MACHINE = "SIM_CPU";

    // Low-level logging backends
    #define BACKEND_LOG_DEBUG(fmt, ...) \
        printf("[DEBUG] %s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)

    #define BACKEND_LOG_INFO(fmt, ...) \
        printf("[INFO] %s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)

    #define BACKEND_LOG_WARN(fmt, ...) \
        printf("[WARN] %s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)

    #define BACKEND_LOG_ERROR(fmt, ...) \
        printf("[ERROR] %s: " fmt "\n", __FUNCTION__, ##__VA_ARGS__)
#endif

// =============================================================================
// Unified High-Level Logging Macros (Platform-Independent Layer)
// =============================================================================

#define D_DEV_LOGD(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_debug()) { \
            BACKEND_LOG_DEBUG(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define D_DEV_LOGI(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_info()) { \
            BACKEND_LOG_INFO(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define D_DEV_LOGW(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_warn()) { \
            BACKEND_LOG_WARN(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

#define D_DEV_LOGE(MODE_NAME, fmt, ...) \
    do { \
        if (is_log_enable_error()) { \
            BACKEND_LOG_ERROR(fmt, ##__VA_ARGS__); \
        } \
    } while(0)

// =============================================================================
// Convenience Macros
// =============================================================================

#define DEV_INFO(fmt, args...)  D_DEV_LOGI(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_WARN(fmt, args...)  D_DEV_LOGW(TILE_FWK_DEVICE_MACHINE, fmt, ##args)
#define DEV_ERROR(fmt, args...) D_DEV_LOGE(TILE_FWK_DEVICE_MACHINE, fmt, ##args)

// =============================================================================
// Platform-Specific Assertion
// =============================================================================

#if defined(__PLATFORM_A2A3__)
    // Real hardware: enable assertions for debugging
    #include <cassert>
    #define DEV_ASSERT(condition) assert(condition)
#else
    // Simulation: disable assertions (graceful error handling)
    #define DEV_ASSERT(condition) ((void)0)
#endif

// =============================================================================
// Conditional Check Macros
// =============================================================================

#define DEV_CHECK_COND_RETURN_VOID(cond, fmt, ...)          \
    do {                                                    \
        if (!(cond)) {                                      \
            DEV_ERROR(fmt, ##__VA_ARGS__);                  \
            DEV_ASSERT(0);                                  \
            return;                                         \
        }                                                   \
    } while(0)

#define DEV_CHECK_COND_RETURN(cond, retval, fmt, ...)       \
    do {                                                    \
        if (!(cond)) {                                      \
            DEV_ERROR(fmt, ##__VA_ARGS__);                  \
            DEV_ASSERT(0);                                  \
            return (retval);                                \
        }                                                   \
    } while(0)

#define DEV_CHECK_POINTER_NULL_RETURN_VOID(ptr, fmt, ...)   \
    do {                                                    \
        if ((ptr) == nullptr) {                             \
            DEV_ERROR(fmt, ##__VA_ARGS__);                  \
            DEV_ASSERT(0);                                  \
            return;                                         \
        }                                                   \
    } while(0)

// =============================================================================
// Helper Functions
// =============================================================================

// Check if log level is enabled (inline for efficiency)
inline bool is_log_enable_debug() { return g_is_log_enable_debug; }
inline bool is_log_enable_info()  { return g_is_log_enable_info; }
inline bool is_log_enable_warn()  { return g_is_log_enable_warn; }
inline bool is_log_enable_error() { return g_is_log_enable_error; }

// Initialize log switch (platform-specific implementation)
#if defined(__PLATFORM_A2A3__)
    void init_log_switch();  // Implemented in device_log.cpp
#else
    inline void init_log_switch() {}  // No-op for simulation
#endif

#endif  // PLATFORM_DEVICE_LOG_H_
