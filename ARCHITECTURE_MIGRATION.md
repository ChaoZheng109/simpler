# Architecture Migration: A2A3 → A5

## Overview

This document describes the architectural restructuring that introduces A5 support while maintaining A2A3 compatibility. The migration establishes a multi-architecture codebase with independent platform and runtime implementations for each architecture.

## Directory Structure Changes

### Before (Single Architecture)
```
src/
├── platform/
│   ├── include/           # Shared headers
│   ├── a2a3/              # Hardware backend
│   ├── a2a3sim/           # Simulation backend
│   └── src/               # Shared implementations
└── runtime/
    ├── host_build_graph/
    ├── aicpu_build_graph/
    └── tensormap_and_ringbuffer/
```

### After (Multi-Architecture)
```
src/
├── a2a3/                  # A2/A3 architecture
│   ├── platform/
│   │   ├── include/       # A2A3-specific headers
│   │   ├── onboard/       # Hardware backend (renamed from a2a3/)
│   │   ├── sim/           # Simulation backend (renamed from a2a3sim/)
│   │   └── src/           # Shared A2A3 implementations
│   └── runtime/
│       ├── host_build_graph/
│       ├── aicpu_build_graph/
│       └── tensormap_and_ringbuffer/
│
└── a5/                    # A5 architecture
    ├── platform/          # ✅ Production-ready
    │   ├── include/       # A5-specific headers
    │   ├── onboard/       # Hardware backend
    │   ├── sim/           # Simulation backend
    │   └── src/           # Shared A5 implementations
    └── runtime/           # ⚠️ Temporary validation runtime
        └── host_build_graph/  # Copied from A2A3 for platform validation
```

## Platform Differences: A2A3 vs A5

### 1. Hardware Capacity

| Configuration | A2A3 | A5 | Change |
|--------------|------|-----|--------|
| Max Block Dimension | 24 | 36 | +50% |
| Max AICPU Threads | 4 | 7 | +75% |
| Max AIC Cores/Thread | 24 | 36 | +50% |
| Max AIV Cores/Thread | 48 | 72 | +50% |
| Max Total Cores/Thread | 72 | 108 | +50% |

**Impact**: A5 provides significantly higher parallelism capacity, requiring updated configuration constants throughout the platform layer.

### 2. Register Communication Protocol

#### Register Offsets
| Register | A2A3 | A5 | Notes |
|----------|------|-----|-------|
| DATA_MAIN_BASE | 0xA0 | 0xD0 | +48 bytes |
| COND | 0x4C8 | 0x5108 | +16KB |
| FAST_PATH_ENABLE | 0x18 | ❌ Removed | Simplified protocol |

#### RegId Enum Changes
- **A2A3**: 3 variants (DATA_MAIN_BASE, COND, FAST_PATH_ENABLE)
- **A5**: 2 variants (DATA_MAIN_BASE, COND)
- **Reason**: Fast path control removed in A5 hardware, simplifying host-device communication

### 3. System Counter Frequency

| Aspect | A2A3 | A5 | Impact |
|--------|------|-----|--------|
| Frequency | 50 MHz | 1000 MHz | 20x faster |
| Conversion | 50M cycles/sec | 1B cycles/sec | Higher profiling precision |
| Nanosecond calculation | `cycles * 20` | `cycles` | Simpler conversion |

**Impact**: A5's 1GHz system counter provides 20x better timing granularity for profiling and performance analysis.

### 4. Simulation Register Mapping

#### Memory Layout Strategy
| Aspect | A2A3 Sim | A5 Sim | Optimization |
|--------|----------|--------|--------------|
| Register block size | 0x500 (1280 bytes) | 0x2000 (8192 bytes) | Sparse mapping |
| Mapping strategy | Contiguous | **Sparse (2 x 4KB pages)** | Memory efficient |
| Page 0 | N/A | 0x0000-0x0FFF | Control registers |
| Page 1 | N/A | 0x5000-0x5FFF | Condition registers |
| Gap (0x1000-0x4FFF) | N/A | Unmapped | Saves 16KB/core |

#### Implementation
**A2A3**: Direct offset calculation
```cpp
volatile uint32_t* reg_ptr = regs_base + offset;
```

**A5**: Sparse mapping helper
```cpp
volatile uint32_t* sparse_reg_ptr(volatile uint32_t* base, uint32_t offset) {
    if (offset < 0x1000) return base + offset / 4;           // Page 0
    if (offset >= 0x5000) return base + 0x400 + (offset - 0x5000) / 4;  // Page 1
    return nullptr;  // Gap region
}
```

**Memory Savings**: For 108 cores, A5 sim saves ~1.7MB compared to contiguous mapping (16KB × 108).

## Runtime Status

### A2A3 Runtime (Production)
- ✅ `host_build_graph` - Host CPU builds full task graph
- ✅ `aicpu_build_graph` - AICPU builds graph on-device
- ✅ `tensormap_and_ringbuffer` - Advanced runtime with tensor maps and ring buffers

### A5 Runtime (Temporary)
- ✅ `host_build_graph` - **Validation runtime only** (copied from A2A3)
- ❌ `aicpu_build_graph` - Not included
- ❌ `tensormap_and_ringbuffer` - Not included

**Strategy**: A5 uses a temporary copy of A2A3's `host_build_graph` to validate the platform layer. Future phases will implement A5-native runtimes optimized for A5 hardware characteristics (higher core counts, simplified protocol, faster counters).

## Build System Changes

### Modified Files
1. **ci.sh** - Multi-architecture runtime discovery
   - Searches both `src/a2a3/runtime/` and `src/a5/runtime/`
   - Supports platform-to-architecture mapping (a2a3→a2a3, a5→a5)

2. **python/runtime_builder.py** - Architecture-aware compilation
   - Resolves runtime paths based on architecture
   - Handles both legacy (`src/runtime/`) and new (`src/{arch}/runtime/`) layouts

3. **python/kernel_compiler.py** - Platform/architecture routing
   - Maps platform names to architecture directories
   - Supports both onboard and sim backends

4. **python/runtime_compiler.py** - Flexible path resolution
   - Detects architecture from runtime path
   - Constructs correct include paths for each architecture

### Platform Mapping
| Platform Name | Architecture | Backend Path |
|--------------|--------------|--------------|
| `a2a3` | a2a3 | `src/a2a3/platform/onboard/` |
| `a2a3sim` | a2a3 | `src/a2a3/platform/sim/` |
| `a5` | a5 | `src/a5/platform/onboard/` |
| `a5sim` | a5 | `src/a5/platform/sim/` |

## File Statistics

### A2A3 Migration (Moved)
- Platform layer: 55 files (10,045 lines)
- Runtime layer: 3 runtimes, 45 files (8,234 lines)
- **Total: 100 files moved from `src/` to `src/a2a3/`**

### A5 Addition (New)
- Platform layer: 55 files (10,045 lines) - **Production-ready**
- Runtime layer: 7 files (2,142 lines) - **Temporary validation code**
- **Total: 62 new files added to `src/a5/`**

### Build System Updates
- Modified: 6 Python scripts and 1 shell script
- Added: 1 test configuration file (`tests/conftest.py`)

## Key Architectural Principles

### 1. Architecture Independence
Each architecture (`a2a3`, `a5`) has completely independent:
- Platform headers with architecture-specific constants
- Hardware and simulation implementations
- Runtime implementations (current or future)

### 2. Platform-Runtime Separation
- **Platform layer**: Hardware abstraction (registers, memory, device control)
- **Runtime layer**: Task scheduling and execution logic
- Runtimes depend on platform headers, but platforms are runtime-agnostic

### 3. Backward Compatibility
- A2A3 code remains unchanged (only moved to new location)
- Existing examples and tests work without modification
- Build system supports both legacy and new path layouts

### 4. Forward Extensibility
- Easy to add new architectures (e.g., `src/a6/`)
- Each architecture can have different runtime variants
- Platform-specific optimizations isolated per architecture

## Testing Strategy

### A2A3 Validation
```bash
# Simulation tests
./ci.sh -p a2a3sim

# Hardware tests
./ci.sh -p a2a3 -d 4-7 --parallel
```

### A5 Validation
```bash
# Simulation tests (using temporary host_build_graph)
./ci.sh -p a5sim

# Hardware tests (using temporary host_build_graph)
./ci.sh -p a5 -d 0-3 --parallel
```

## Migration Checklist

- [*] Restructure A2A3 code into `src/a2a3/`
- [*] Create A5 platform layer with updated constants
- [*] Implement A5 sparse register mapping for simulation
- [*] Add temporary A5 runtime for validation
- [*] Update build system for multi-architecture support
- [*] Update CI scripts for architecture discovery
- [*] Add test configuration for architecture selection
- [*] Validate A5 platform on simulation
- [ ] Validate A5 platform on hardware
- [ ] Design A5-native runtime architecture
- [ ] Implement A5-native runtimes

## Future Work

### Phase 3: A5-Native Runtime Development
1. **Design**: Create runtime architecture optimized for A5 characteristics
   - Leverage 50% higher core counts
   - Optimize for simplified register protocol (no fast path)
   - Utilize 20x faster system counter for fine-grained profiling

2. **Implementation**: Develop A5-specific runtimes
   - `host_build_graph` - A5-optimized host-side graph builder
   - `aicpu_build_graph` - A5-optimized on-device scheduler
   - `tensormap_and_ringbuffer` - Advanced A5 runtime with new features

3. **Optimization**: A5-specific performance tuning
   - Memory management for 108-core configurations
   - Scheduling strategies for 7-thread AICPU
   - Synchronization primitives for sparse register layout

### Phase 4: Multi-Architecture Testing
- Unified test suite running on both A2A3 and A5
- Performance comparison framework
- Architecture-specific benchmarks
- Regression testing across architectures

## References

- [src/a2a3/platform/include/common/platform_config.h](src/a2a3/platform/include/common/platform_config.h) - A2A3 configuration
- [src/a5/platform/include/common/platform_config.h](src/a5/platform/include/common/platform_config.h) - A5 configuration
- [src/a5/runtime/README.md](src/a5/runtime/README.md) - A5 runtime status
- [ci.sh](ci.sh) - Multi-architecture CI script
- [python/runtime_builder.py](python/runtime_builder.py) - Architecture-aware build system
