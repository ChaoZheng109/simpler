/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file cpu_sim_state.h
 * @brief Internal header for CPU simulation state lifecycle management
 *
 * Declares clear_cpu_sim_shared_storage() for DeviceRunner to call at
 * run() entry and finalize() to reset simulation state between runs.
 */

#ifndef PLATFORM_A5SIM_HOST_CPU_SIM_STATE_H_
#define PLATFORM_A5SIM_HOST_CPU_SIM_STATE_H_

void clear_cpu_sim_shared_storage();

#endif  // PLATFORM_A5SIM_HOST_CPU_SIM_STATE_H_
