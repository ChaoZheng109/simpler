# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile topology consumers against old and public driver declarations without CANN."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_headers(root: Path, kind: str | None) -> None:
    toolkit = root / "cann/include/driver"
    driver = root / "driver/include"
    toolkit.mkdir(parents=True, exist_ok=True)
    driver.mkdir(parents=True, exist_ok=True)
    (toolkit / "dsmi_common_interface.h").write_text("#pragma once\n")
    hal = toolkit / "ascend_hal_base.h"
    dsmi = driver / "dsmi_common_interface.h"
    if kind is None:
        hal.write_text("#pragma once\n")
        dsmi.unlink(missing_ok=True)
        return
    if kind == "enum":
        hal.write_text("#pragma once\nenum { INFO_TYPE_CPU_TOPO = 73 };\n")
        selector = "enum { DSMI_SOC_INFO_SUB_CMD_CPU_TOPO = 11 };"
    else:
        hal.write_text("#pragma once\n#define INFO_TYPE_CPU_TOPO 73\n")
        selector = "#define DSMI_SOC_INFO_SUB_CMD_CPU_TOPO 11"
    dsmi.write_text(
        "#pragma once\n"
        + selector
        + "\n#define DSMI_MAX_CPU_TOPO_NUM 12\n"
        + "struct dsmi_single_cpu_topology_info { unsigned long long cpu_mask; };\n"
        + "struct dsmi_cpu_topology_info {\n"
        + "  unsigned int total_nums;\n"
        + "  dsmi_single_cpu_topology_info single_cpu_topo_info[DSMI_MAX_CPU_TOPO_NUM];\n"
        + "};\n"
    )


def _build_consumer(root: Path, *, public: bool) -> None:
    if not shutil.which("cmake"):
        pytest.skip("cmake is required for the driver header compile test")
    (root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(topology_header_test LANGUAGES CXX)\n"
        "set(CMAKE_CXX_STANDARD 17)\n"
        f'include("{REPO_ROOT.as_posix()}/cmake/aicpu_topology_driver.cmake")\n'
        "add_executable(consumer consumer.cpp)\n"
        "simpler_configure_aicpu_topology_driver(consumer)\n"
        'target_include_directories(consumer PRIVATE "${ASCEND_HOME_PATH}/include")\n'
    )
    hal, dsmi, capacity = (73, 11, 12) if public else (59, 2, 64)
    assertions = (
        "static_assert(std::is_same_v<pto::driver::CpuTopology, dsmi_cpu_topology_info>);\n"
        "static_assert(std::is_same_v<pto::driver::SingleCpuTopology, dsmi_single_cpu_topology_info>);\n"
        if public
        else "static_assert(sizeof(pto::driver::SingleCpuTopology) == 16);\n"
        "static_assert(sizeof(pto::driver::CpuTopology) == 1032);\n"
    )
    (root / "consumer.cpp").write_text(
        '#include "aicpu_topology_driver.h"\n'
        f"static_assert(pto::driver::kCpuTopoHalInfoType == {hal});\n"
        f"static_assert(pto::driver::kCpuTopoDsmiSubcommand == {dsmi});\n"
        f"static_assert(pto::driver::kCpuTopoCapacity == {capacity});\n" + assertions + "int main() {}\n"
    )
    for command in [
        [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(root / "build"),
            f"-DASCEND_HOME_PATH={root / 'cann'}",
            f"-DASCEND_DRIVER_PATH={root / 'driver'}",
        ],
        ["cmake", "--build", str(root / "build")],
    ]:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["macro", "enum"])
def test_driver_declarations_override_legacy_toolkit(tmp_path: Path, kind: str) -> None:
    _write_headers(tmp_path, kind)
    _build_consumer(tmp_path, public=True)


def test_reconfigure_refreshes_driver_capabilities(tmp_path: Path) -> None:
    for kind in (None, "enum", None):
        _write_headers(tmp_path, kind)
        _build_consumer(tmp_path, public=kind is not None)
