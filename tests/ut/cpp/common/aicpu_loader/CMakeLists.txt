# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Test dispatcher path selection and publication with a CPU-only CANN context stub.
macro(aicpu_loader_case src)
    simpler_ut_plain_case(${src}
        SOURCES ${SIMPLER_SRC}/common/aicpu_loader/device/aicpu_dispatcher.cpp
                ${SIMPLER_SRC}/common/platform/sim/aicpu/cache_ops.cpp
        INCLUDES ${SIMPLER_SRC}/common ${SIMPLER_SRC}/common/platform/include
                 ${CMAKE_CURRENT_SOURCE_DIR}/support LABEL no_hardware)
endmacro()
simpler_ut_glob_cases(aicpu_loader_case)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(test_so_upload PRIVATE "LINKER:--wrap=write" "LINKER:--wrap=statvfs")
endif()
