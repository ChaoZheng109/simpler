"""
PTO Runtime ctypes Bindings

Provides a Pythonic interface to the PTO runtime via ctypes.
Users must provide a pre-compiled libpto_runtime.so (built via binary_compiler.py).

Usage:
    from runtime_bindings import load_runtime

    DeviceRunner, Graph = load_runtime("/path/to/libpto_runtime.so")
    runner = DeviceRunner()
    runner.init(device_id=0, aicpu_binary=aicpu_bytes, aicore_binary=aicore_bytes, pto_isa_root="/path/to/pto-isa")

    graph = Graph()
    graph.initialize()

    runner.run(graph, block_dim=3, launch_aicpu_num=1)
    runner.print_handshake_results(graph)
    graph.validate_and_cleanup()
    runner.finalize()
"""

from ctypes import (
    CDLL,
    POINTER,
    c_int,
    c_void_p,
    c_uint8,
    c_size_t,
    c_char_p,
)
from pathlib import Path
from typing import Union
import ctypes
import tempfile


# ============================================================================
# Runtime Library Loader
# ============================================================================

class RuntimeLibraryLoader:
    """Loads and manages the PTO runtime C API library."""

    def __init__(self, lib_path: Union[str, Path]):
        """
        Load the PTO runtime library.

        Args:
            lib_path: Path to libpto_runtime.so

        Raises:
            FileNotFoundError: If library file not found
            OSError: If library cannot be loaded
        """
        lib_path = Path(lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib_path = lib_path
        self.lib = CDLL(str(lib_path))
        self._setup_functions()

    def _setup_functions(self):
        """Set up ctypes function signatures."""
        # Graph functions
        self.lib.InitGraph.argtypes = [c_void_p]
        self.lib.InitGraph.restype = c_int

        self.lib.ValidateGraph.argtypes = [c_void_p]
        self.lib.ValidateGraph.restype = c_int

        # DeviceRunner functions
        self.lib.DeviceRunner_Init.argtypes = [c_int,
                                               POINTER(c_uint8), c_size_t,
                                               POINTER(c_uint8), c_size_t,
                                               c_char_p]
        self.lib.DeviceRunner_Init.restype = c_int

        self.lib.DeviceRunner_Run.argtypes = [c_void_p, c_int, c_int]
        self.lib.DeviceRunner_Run.restype = c_int

        self.lib.DeviceRunner_PrintHandshakeResults.argtypes = [c_void_p]
        self.lib.DeviceRunner_PrintHandshakeResults.restype = None

        self.lib.DeviceRunner_Finalize.argtypes = []
        self.lib.DeviceRunner_Finalize.restype = c_int


# ============================================================================
# Python Wrapper Classes
# ============================================================================

class Runtime:
    """
    Task dependency runtime.

    Python wrapper around the C Runtime API.
    Runtimes are allocated and managed by C++. Python just holds handles.
    """

    def __init__(self, lib: CDLL):
        """
        Create a new runtime handle.

        Args:
            lib: Loaded ctypes library (RuntimeLibraryLoader.lib)
        """
        self.lib = lib
        # Allocate buffer to hold a Runtime pointer (8 bytes on 64-bit)
        # This buffer will be filled by C++ during InitGraph()
        self._buffer = ctypes.create_string_buffer(ctypes.sizeof(c_void_p))
        # Get the address of this buffer to pass to C++
        # C++ will dereference this as Runtime** and fill in the Runtime*
        self._handle = ctypes.addressof(self._buffer)

    def initialize(self) -> None:
        """
        Initialize the runtime structure.

        Calls InitGraph() in C++ which allocates the Runtime, builds tasks,
        allocates device tensors, initializes data, and runs the runtime.

        Raises:
            RuntimeError: If initialization fails
        """
        rc = self.lib.InitGraph(self._handle)
        if rc != 0:
            raise RuntimeError(f"Failed to initialize Runtime: {rc}")

    def validate_and_cleanup(self) -> None:
        """
        Validate results and cleanup all resources.

        Calls ValidateGraph() in C++ which validates computation results,
        frees device tensors, and deletes the runtime.

        Raises:
            RuntimeError: If validation fails
        """
        # Get the Runtime* pointer that was stored in the buffer by InitGraph
        # _buffer contains [Runtime*], so cast it to pointer to void* and dereference
        runtime_ptr = ctypes.cast(self._buffer, ctypes.POINTER(c_void_p)).contents
        rc = self.lib.ValidateGraph(runtime_ptr)
        if rc != 0:
            raise RuntimeError(f"ValidateGraph failed: {rc}")

    def __del__(self):
        """Clean up runtime resources."""
        # Runtime is managed by C++ (deleted in ValidateGraph), no explicit cleanup needed
        pass


class DeviceRunner:
    """
    Device execution runtime.

    Python wrapper around the C DeviceRunner API.
    Supports context manager protocol for automatic cleanup.
    """

    def __init__(self, lib: CDLL):
        """
        Initialize the DeviceRunner wrapper.

        Args:
            lib: Loaded ctypes library (RuntimeLibraryLoader.lib)
        """
        self.lib = lib
        self._initialized = False

    def init(
        self,
        device_id: int,
        aicpu_binary: bytes,
        aicore_binary: bytes,
        pto_isa_root: str,
    ) -> None:
        """
        Initialize the device runner.

        Args:
            device_id: Device ID (0-15)
            aicpu_binary: Binary data of AICPU shared object
            aicore_binary: Binary data of AICore kernel
            pto_isa_root: Path to PTO-ISA root directory (headers location)

        Raises:
            RuntimeError: If initialization fails
        """
        # Convert bytes to ctypes arrays
        aicpu_array = (c_uint8 * len(aicpu_binary)).from_buffer_copy(aicpu_binary)
        aicore_array = (c_uint8 * len(aicore_binary)).from_buffer_copy(aicore_binary)

        rc = self.lib.DeviceRunner_Init(
            device_id,
            aicpu_array,
            len(aicpu_binary),
            aicore_array,
            len(aicore_binary),
            pto_isa_root.encode('utf-8'),
        )
        if rc != 0:
            raise RuntimeError(f"DeviceRunner_Init failed: {rc}")
        self._initialized = True

    def run(self, runtime: "Runtime", block_dim: int, launch_aicpu_num: int = 1) -> None:
        """
        Execute a runtime on the device.

        Args:
            runtime: Runtime to execute (must have been initialized via runtime.initialize())
            block_dim: Number of blocks (1 block = 1 AIC + 2 AIV)
            launch_aicpu_num: Number of AICPU instances

        Raises:
            RuntimeError: If execution fails
        """
        if not self._initialized:
            raise RuntimeError("DeviceRunner not initialized")

        # Get the actual Runtime* pointer from the buffer
        runtime_ptr = ctypes.cast(runtime._buffer, ctypes.POINTER(c_void_p)).contents
        rc = self.lib.DeviceRunner_Run(runtime_ptr, block_dim, launch_aicpu_num)
        if rc != 0:
            raise RuntimeError(f"DeviceRunner_Run failed: {rc}")

    def print_handshake_results(self, runtime: "Runtime") -> None:
        """
        Print handshake results from device.

        Args:
            runtime: Runtime whose handshake results should be printed

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("DeviceRunner not initialized")

        # Get the actual Runtime* pointer from the buffer
        runtime_ptr = ctypes.cast(runtime._buffer, ctypes.POINTER(c_void_p)).contents
        self.lib.DeviceRunner_PrintHandshakeResults(runtime_ptr)

    def finalize(self) -> None:
        """Cleanup all resources."""
        if self._initialized:
            rc = self.lib.DeviceRunner_Finalize()
            if rc != 0:
                raise RuntimeError(f"DeviceRunner_Finalize failed: {rc}")
            self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        try:
            self.finalize()
        except Exception:
            pass  # Ignore errors during cleanup
        return False

    def __del__(self):
        """Clean up on deletion."""
        try:
            if self._initialized:
                self.finalize()
        except Exception:
            pass  # Ignore errors during cleanup


# ============================================================================
# Public API
# ============================================================================

def load_runtime(lib_path: Union[str, Path, bytes]) -> tuple:
    """
    Load the PTO runtime library and return wrapper classes.

    Args:
        lib_path: Path to libpto_runtime.so (str/Path), or compiled binary data (bytes)

    Returns:
        Tuple of (DeviceRunnerClass, RuntimeClass) initialized with the library

    Example:
        # From file path
        DeviceRunner, Runtime = load_runtime("/path/to/libpto_runtime.so")

        # From compiled binary bytes
        host_binary = compiler.compile("host", include_dirs, source_dirs)
        DeviceRunner, Runtime = load_runtime(host_binary)

        runner = DeviceRunner()
        runner.init(device_id=0, aicpu_binary=aicpu_bytes, aicore_binary=aicore_bytes, pto_isa_root="/path/to/pto-isa")

        runtime = Runtime()
        runtime.initialize()

        runner.run(runtime, block_dim=3, launch_aicpu_num=1)
        runner.print_handshake_results(runtime)
        runtime.validate_and_cleanup()
        runner.finalize()
    """
    # If bytes are provided, write to temporary file
    if isinstance(lib_path, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.so') as f:
            f.write(lib_path)
            lib_path = f.name

    loader = RuntimeLibraryLoader(lib_path)
    lib = loader.lib

    # Create wrapper classes with the loaded library
    class _DeviceRunner(DeviceRunner):
        def __init__(self):
            super().__init__(lib)

    class _Runtime(Runtime):
        def __init__(self):
            super().__init__(lib)

    return _DeviceRunner, _Runtime
