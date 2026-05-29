# A5 核内泳道图实施方案

把核内泳道（L0 swimlane）在 simpler A5 onboard 上落地的工程方案。**只读本文
之前**先看一遍 [core_swimlane_principle.md](core_swimlane_principle.md)——
本文不重复硬件原理与数据 chunk 格式，只讨论"怎么落地到本项目"。

适用范围：A5 onboard 平台（dav-c310）。sim 不在本文范围。

> **架构修订 (2026-05-25)：consumer 从 AICPU 上移到 host**
>
> 第一版方案让 AICPU scheduler 线程直接调 `prof_drv_start` /
> `prof_channel_read` 当 consumer（原 §4.1）。**实测验证不可行**：
>
> 1. AICPU 端 `/usr/lib64/libascend_hal.so` device-stub 的 `prof_drv_start` 在
>    错的 `device_id` 上 hang，在对的 `device_id` 上明确返
>    **`DRV_ERROR_NOT_SUPPORT (0xFFFE)`**——driver 主动拒绝 AICPU 当
>    biu_perf consumer。Host 一侧 driver lib (3.05 MB) 同一调用返 0。
> 2. host 与 AICPU 上的 `libascend_hal.so` 是**两份不同的二进制**：host 是 PCIe
>    用户态驱动（完整 ioctl 链路），AICPU 是 device rootfs 里的 38 KB stub，
>    biu_perf consumer API 在 stub 里没真实现。
> 3. 项目其它 profiling 子系统（L2 / PMU / dump）在 AICPU 端**完全不调任何
>    driver 接口**（纯 GM 指针 + `rtMemcpy` 回流）——L0 是第一个想用
>    AICPU driver consumer 的子系统，直接踩到 stub 完成度天花板。
>
> 新架构：**host 端起一个 ProfilerBase derived collector（`L0PerfCollector`）
> 调 `prof_drv_start` + `prof_channel_read`，AICPU 只在 task FIN 后向 GM
> ReadyQueue push 一个 `L0TaskFinMarker{task_id, group}`。** Marker 在
> `on_buffer_collected(marker)` 里触发 host 这一侧 drain 对应 3 个 sub-core
> channel，把解码后的 stamp 归到 `marker.task_id`。详见 §3 新架构图与
> §11 实施细化。
>
> **验证基线说明**：原理文档背景基于 **CANN 9.0.0（version 26.0.rc1）**；
> 当前落地环境为 **CANN 9.1.T500**，头文件路径未变（`asc/include/basic_api/...`、
> `include/driver/ascend_hal_base.h` 都在原位）。
>
> - §6 **TODO 2**：✅ 已在 9.1.T500 上验证（`user_data` 16 字节结构、
>   `aicore_phys_id {0,9,17,18,27,35}`、`prof_start_para` 字段值）。
> - §6 **TODO 1**：channel_id 11..28 映射在 9.0.0 上验证过，9.1.T500
>   上**不复跑**（假设一致；如落地后看到符号不对再回头查）。
> - §6 **TODO 5**：✅ AICPU NOT_SUPPORT 已实测（2026-05-25 dev=2 + 4 张卡复现）。
> - 原理文档里 `__NPU_ARCH__ == 3510`、MTE 自动 `pipe_barrier` 规则、
>   4 字节 chunk 结构这些"硬件行为类"结论同样建议在 9.1.T500 上复读一次
>   头文件 / 跑一次 stamp 解码做佐证。
>
> **a5 profiling 框架基线（#777 之后）**：a5 已经对齐 a2a3，所有现存子
> 系统（L2Perf / PMU / TensorDump）继承公共 `ProfilerBase<Derived, Module>`
> CRTP，profiling 使能位通过 `KernelArgs::enable_profiling_flag` bitmask
> 传到 AICPU，AICore 用 stable per-core staging ring 解耦 AICPU 的 buffer
> 轮换。L0 perf 的设计**必须按这套框架**写，本文 §4 / §8 / §11 全部按此对仗。

---

## 1. 目标

- 在 simpler runtime 内置一条 **不依赖 msprof daemon** 的核内泳道采集通路
- **per-stamp 数据归属到具体 task** —— 通过 AICPU 在每个 task FIN 后向 GM
  ReadyQueue push 的 `L0TaskFinMarker{task_id, group}` 严格按 FIN 顺序
  传递任务边界，host consumer 顺次 drain channel、把 chunk 归到对应 marker
- 完整复用现有 host↔device profiling 基础设施
  ([`L2PerfCollector`](../include/host/l2_perf_collector.h) /
  [`PmuCollector`](../include/host/pmu_collector.h) /
  [`profiling-framework.md`](../../../../docs/profiling-framework.md))，
  增量小、不动核心调度路径
- 用户 kernel 写 stamp 的方式与项目现存 kernel 一致——当前直接用
  `bisheng::cce::mark_stamp<PIPE_X, id>()` 内建（见
  [`kernel_add_scalar.cpp`](../../../../examples/a5/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp)
  现状）；后续可平滑切换到 CANN 公开包装 `AscendC::MarkStamp`（仅图升级
  抗性，运行期等价，详见 §4.3 末尾）

---

## 2. 选型对比

下表把所有可行路径列齐，**采用第 4 行**（C 的 host-consumer 变种——原 C 的
AICPU consumer 实测被 driver 拒）。

| 方案 | stamp 数据源 | 谁是 consumer | per-task 归属 | 依赖 msprof CLI | GM 带宽 | 评价 |
| ---- | ------------ | ------------- | ------------- | --------------- | ------- | ---- |
| A. msprof CLI 包住进程 | HW `__dfx_region` | msprof daemon | ❌ 一次 launch = 一整个 runtime，stamp 全混在一起 | ✅ 必须 | 0 | 跑通最快，但 timeline 无法分 task |
| B. AICore 写 GM（软件 stamp） | `get_sys_cnt()` + 写 GM | AICPU 调度线程 | ✅ AICPU 在 task 入口注入槽位 | ❌ | 16B/stamp | 工程上最干净，但放弃 HW 通路的零 GM 带宽优势 |
| ~~C. HW stamp + AICPU scheduler 线程当 consumer~~ | HW `__dfx_region` | ~~AICPU scheduler 线程~~ | ✅ | ❌ | 0 | ❌ **被 driver 拒**：AICPU `prof_drv_start` 返 `DRV_ERROR_NOT_SUPPORT` (§6 TODO 5) |
| **C'. HW stamp + host worker 当 consumer + AICPU marker 上报**（**采用**） | HW `__dfx_region` | host `L0PerfCollector` (ProfilerBase derived)；AICPU FIN 后 push `L0TaskFinMarker{task_id, group}` 到 ReadyQueue，host 顺次 drain 对应 3 个 sub-core channel 并归到 `marker.task_id` | ✅ marker 严格按 FIN 顺序传递任务边界，host 按 marker 顺次 drain 自然保留归属 | ❌ | 0（HW DFX bus + driver 走的是 PCIe ioctl，不占 GM） | 兼得 HW 零 GM 带宽 + per-task 归属；driver consumer 在 host driver lib 上完整可用；与 PMU/L2 框架同形 |

**为什么不选 A**：simpler A5 一次 host launch 跑完整个 runtime，msprof
的"按 launch 区间染色"机制在我们这里粒度太粗，timeline 看不出任何 op 边界。

**为什么不选 B**：HW DFX bus 与 GM 独立，密集打点（每 task 几十条
stamp）对真实数据访问的 L2 命中率零影响；且 `bisheng::cce::mark_stamp` /
`AscendC::MarkStamp` 都是 CANN 自带内建，不引入项目自有头封装。

**为什么从 C 改成 C'**（见顶部修订块）：CANN 9.1.T500 AICPU device-stub 的
`prof_drv_start` 在正确 `device_id` 下返 `DRV_ERROR_NOT_SUPPORT (0xFFFE)`——driver
明确拒绝 AICPU 当 biu_perf consumer；host 一侧同样调用 (dev=2) 返 0。
跨进程的 PCIe driver 路径在 host 是完整实现，只能在 host 跑 consumer。

**C' 与 C 的本质差**：

| 维度 | C（被拒） | C'（采用） |
| --- | --- | --- |
| 谁 `prof_drv_start` | AICPU scheduler 线程 (NOT_SUPPORT) | host collector `initialize()` (返 0) |
| 谁 `prof_channel_read` | AICPU scheduler 线程，task FIN 同步 read | host collector `on_buffer_collected(marker)` |
| task_id 归属载体 | scheduler 局部变量直接拿 | AICPU 写 marker 到 GM ReadyQueue，host 顺次 pop |
| 反压 | scheduler 卡 → AICCore 等 dispatch | mgmt 拉 marker + collector drain，框架 SPSC 自吸收 |
| 走的 lib | AICPU `/usr/lib64/libascend_hal.so` (38 KB stub，broken) | host `/usr/local/Ascend/driver/lib64/driver/libascend_hal.so` (3 MB 完整实现) |

---

## 3. 整体架构

> 关键事实：A5 一颗 SoC 有 **36 个 AICore（每核 = 1 cube + 2 vector，又叫
> cluster / blockdim）**，但 **biu_perf 硬件采集当前只覆盖 6 个 AICore**
> （详见 §10 已知硬件限制）。所以一次 run 只对前 6 个 AICore 出核内泳道；
> block_dim > 6 时超出部分的 task **没有 stamp 数据**。

```text
┌─────── DEVICE (dav-c310 SoC) ───────────────────────────────────────────┐
│                                                                          │
│  AICore #N 的物理 sub-core（每核 1 aic + 2 aiv，6 核共 18 sub-core）       │
│    user kernel:  bisheng::cce::mark_stamp<PIPE_V, region_id>();          │
│         │ inline 展开 → __dfx_region 硬件指令 (见原理文档 §3)              │
│         ▼                                                                │
│    HW DFX trace bus ──► SoC trace ring buffer ×18 (per sub-core, 4 MiB)  │
│      (driver 管，host 通过 prof_channel_read 拉)                          │
│                                                                          │
│  AICPU scheduler 线程 #T                                                  │
│    init: 建 logical_core_id → biu_perf group (0..5 / -1) 映射表           │
│          (一次性，不调任何 driver 接口)                                   │
│                                                                          │
│    每跑完一个 task (融进 scheduler 状态机):                                │
│      launch task T on this AICore                                        │
│      wait FIN                                                            │
│      group = s_l0_logical_to_group[core_id]                              │
│      if group >= 0:                                                      │
│         L0TaskFinMarker m{task_id=T.id, group}                           │
│         enqueue_l0_marker(thread_idx, m)                                 │
│            ─── push 到 GM ReadyQueue (现有 ProfilerBase SPSC)            │
│      继续 launch 下一个 task                                              │
│                                                                          │
└─────────────────────┬────────────────────────────────────────────────────┘
                      │ GM SPSC ReadyQueue (现有 ProfilerBase 机制)
                      ▼
┌─────── HOST ────────────────────────────────────────────────────────────┐
│  L0PerfCollector : ProfilerBase<L0PerfCollector, L0PerfModule>          │
│                                                                          │
│    initialize(dev_id):                                                   │
│      hal_handle = dlopen("libascend_hal.so", RTLD_NOW|RTLD_LOCAL)        │
│      resolve prof_drv_start / prof_channel_read / prof_stop              │
│      for chan in 11..28 (18 biu_perf channels):                          │
│         prof_drv_start(dev_id, chan, &biu_user_data)                     │
│              ↑ 已验证 dev=2 healthy 时返 0 (host driver lib 完整实现)     │
│                                                                          │
│    mgmt_thread (框架自带):                                                │
│      每 10μs tick: mirror_shm_from_device                                │
│      扫 AICPU per-thread ready queue → 取 L0TaskFinMarker                │
│      push 到 host manager_.ready_queue                                   │
│                                                                          │
│    collector_thread (框架自带, wait_pop_ready):                          │
│      on_buffer_collected(marker):                                       │
│        for sub in 0..2:                                                  │
│          chan = biu_perf_chan_id(marker.group, sub)                      │
│          while bytes = prof_channel_read(dev_id, chan, buf, sz) > 0:     │
│            l0_decode_chunks(buf, bytes) → emit (cycle, pipe, region_id)  │
│            push L0PerfRecord{                                            │
│              task_id=marker.task_id, sub_core_id, cycle, pipe, region_id │
│            }                                                             │
│                                                                          │
│    finalize: prof_stop 所有 channel + dlclose +                          │
│              写 outputs/<case>/l0_perf_records.json                      │
│                                                                          │
│  swimlane_converter.py                                                   │
│    读 l2_perf_records.json + l0_perf_records.json                        │
│    把每个 task 的横条下面挂 per-pipe 子 lane                               │
└──────────────────────────────────────────────────────────────────────────┘
```

**架构关键点**：

- **AICPU 不再调任何 driver consumer API**——只做"FIN 后 push marker"这个
  GM 写动作，几十纳秒级开销。比原 C 方案的 AICPU 端代码量少 ~80%
- **task_id 归属由 marker 携带**：marker 在 ReadyQueue 里严格按 FIN 顺序，
  host 顺次 pop + drain 自然保留归属（详见 §4.1 顺序论证）
- **driver 调用全在 host 一个线程**：`prof_drv_start` (initialize) +
  `prof_channel_read` (on_buffer_collected) + `prof_stop` (finalize) 都在
  `collector_thread` 上跑，没有"channel 按调用线程归属"问题（同进程同线程）
- **复用 ProfilerBase 两线程**：`mgmt_thread` 拉 marker、`collector_thread`
  在 `on_buffer_collected` 里 drain，**0 个新增常驻线程**，与 PMU/L2 同形
- **没有 task 边界外的 stamp**：marker 顺次 pop + 每个 marker 后 drain 当前
  group 的 3 个 channel，HW ring 里属于该 task 的 chunk 在这个时间窗被独占
  拉走（前提是 §4.2 兜底就位）

## 4. 关键设计点

### 4.1 AICPU scheduler 线程的核内泳道扩展

**AICPU 端不调任何 driver consumer API**。每个 scheduler 线程在它管的
AICore 上跑完一个 task（收到 FIN）后，做一件事：往 GM ReadyQueue push 一
个 `L0TaskFinMarker{task_id, group}`。后续 channel drain 全部由 host
`L0PerfCollector` 在 `on_buffer_collected` 里完成。

关键职责（按 scheduler 线程的生命周期顺序）：

1. **init 阶段——只建映射表**：每个 scheduler 线程在拿到自己的
   `cores_owned[]` 之后，调 `l0_perf_aicpu_init(thread_idx, cores_owned,
   phys_ids, core_num)`。该函数内部仅做一件事：把每个 logical core id 经
   `l0_perf_phys_to_group(phys_id)` 转成 biu_perf group (0..5 或 -1)，写入
   `s_l0_logical_to_group[]`。**没有任何 driver 调用，永不失败**。
2. **task FIN 后——push marker**：scheduler 状态机的"FIN → launch
   下一个 task"之间插入：
   ```cpp
   void l0_perf_aicpu_drain_after_fin(int core_id, int thread_idx, uint64_t task_id) {
       int group = s_l0_logical_to_group[core_id];
       if (group < 0) return;                              // 核不在 biu_perf 覆盖内
       L0TaskFinMarker m{task_id, (uint32_t)group, 0};
       enqueue_l0_marker_ready(thread_idx, m);             // GM SPSC, 几十 ns
   }
   ```
   函数名沿用 `drain_after_fin` 是为了 scheduler 集成点不变；语义已经从
   "drain"变成"signal"。
3. **GM 数据结构**：复用 `ProfilerBase<Derived, Module>` 的 `ReadyQueue` +
   per-instance `FreeQueue` 机制，**ReadyEntry payload 就是
   `L0TaskFinMarker`**（小于一个 cache line，不需要单独 buffer 分配）。
   `L0PerfRecord[]` 仍然是最终落盘的数据结构，但**写入者从 AICPU 变成
   host collector**；AICPU 端没有 `L0PerfBuffer` rotating/free queue 的全
   套机制——marker 自身就是 entry payload，无需 buffer 池。
4. **shutdown 阶段——什么都不做**：AICPU 端没有要 `prof_stop` 的 channel。
   `flush_buffers` 接口保留为 no-op（或干脆删，看 scheduler 集成点是否还在
   调）。host collector 在 `finalize()` 里 `prof_stop` 所有 channel。

**与原 C 方案（AICPU consumer）相比的取舍**：

| 维度 | 原 C 方案（已弃） | C' 方案（采用） |
| --- | --- | --- |
| AICPU 端职责 | dlsym + prof_drv_start + prof_channel_read + chunk decode + L0PerfBuffer 轮换 | 仅 push 一个 16 字节 marker |
| AICPU 端代码量 | ~400 行 | ~50 行 |
| driver 调用方 | AICPU device-stub (broken, NOT_SUPPORT) | host driver lib (3 MB, 完整) |
| FIN→stamp 延迟 | scheduler 串行 drain，几 μs | host collector 异步 drain，~10-100 μs |
| 反压模式 | scheduler 卡 → AICore dispatch 等 | 框架 SPSC 自吸收 (mgmt + collector 两段) |
| 与 PMU/L2 对仗 | 不对仗（AICPU 当 consumer 是异类） | 完全对仗（host 当 consumer，与现存 collector 同形） |

**Marker 顺序 vs Channel drain 时序的论证**（per-task 归属的核心）：

scheduler 线程跑 task A → emit marker_A → 跑 task B → emit marker_B，
两个 marker 在 ReadyQueue 中严格按这个顺序排列（SPSC 不会乱序）。host
mgmt 把它们顺次拉到 host 内部 ready_queue，collector_thread 顺次 pop。

- pop marker_A → 调 `prof_channel_read(group_A 的 3 个 sub-core)` → HW
  ring 里 task A 跑过程中写入的 chunk 被一次性拉走 → 归到 task_A
- pop marker_B → 调 `prof_channel_read(group_B 的 3 个 sub-core)` → HW
  ring 里 task B 跑过程中写入的 chunk 被拉走 → 归到 task_B

**风险点**：marker_A 入 ReadyQueue 时刻 ≠ task A 的 chunk 完全落 HW ring
的时刻。FIN 到达 AICPU 时，AICore 可能还有少量"飞行中"的 trace chunk 没
从 DFX bus 落到 SoC ring（与原 C 方案是同一个硬件假设）。原 C 方案用
§4.2 三层兜底吸收，C' 方案完全沿用——只是 retry loop 从 AICPU 搬到了
host `on_buffer_collected` 里。

**延迟放大风险**：mgmt 拉 marker → collector pop → 调 driver drain 的
端到端延迟通常在 10-100 μs 量级，比原 C 方案 AICPU 同步 drain 的几 μs
长 1-2 个数量级。若 marker_A 处理延迟到 task B 已经在跑、HW ring 已经混
入 task B 的 chunk，drain marker_A 时会把 task B 的 chunk 也拉走、错误归
为 task A。缓解：

1. **kernel `bar.all`**（§4.2 第 1 层，**强烈推荐**用户 kernel 末尾插）
   保证 task A 的 trace flush 与 FIN 同步。
2. mgmt 10μs tick + collector 阻塞唤醒 → 端到端在低负载下 ~50μs，远小于
   一般 task 周期。
3. 若实际跑出顺序错位，加 §4.2 第 3 层 sentinel region_id 4095 作 debug
   断言。

### 4.2 task 边界 = drain 时机（三层兜底）

> **2026-05-26 实测更新**：driver 走"事件触发 batch flush"模型而不是流式：
> 同一 group 的多个 marker drain 中，**只有第一个 marker** 拿到全量数据
> （典型 200KB–1MB），后续 marker 读返 0（ring 已被掏空）。这意味着原版
> §4.2 担心的"飞行中 chunk 漏抓"在实测中没出现——marker 入队那一刻数据
> 已经在 driver buffer 里成块就绪。下面"三层兜底"中第 1、2 条仍然有意义
> （barrier 让 chunk 入 ring 的时机更确定 + retry 兼容多次 read），第 3
> 条 sentinel 已实测不再需要，仅作调试备选。

C' 方案下"task 边界"由 marker 在 ReadyQueue 中的顺序决定。host collector
顺次 pop + drain 时，希望"drain 当前 marker 时 HW ring 里只有该 task 的
chunk"。这要求 trace chunk 在 marker 入队前已经从 DFX bus flush 到 SoC
ring。trace bus 是与数据 pipe 同量级的硬件流水，**FIN 之前 chunk 已发射
但未到达 ring 的可能性存在**，按代价升序的三层兜底：

1. **【强烈推荐 / 用户面约定】kernel 末尾插 `pipe_barrier(PIPE_ALL)`**（或
   等价的 `asm volatile("bar.all")`，`kernel_add_scalar.cpp:112` 就是这写
   法）。barrier 把 V/M/MTE/FIX 全排空再触发 FIN，trace bus 延迟与数据
   pipe 同量级，等到数据 pipe 都排空时，trace 写入 SoC ring 的概率压倒
   性收敛。代价：barrier 串行化已经写完的"末段并行"，但 task 末尾本来就
   要 wait 完所有 pipe，扰动可忽略。**C' 方案下 host drain 延迟远大于
   AICPU 同步 drain，这条 barrier 比原 C 方案更关键**。
2. **【runtime 默认兜底】`prof_channel_read` 有界 retry**：collector 的
   `on_buffer_collected(marker)` 里，对该 group 的 3 个 channel 各做有界
   retry——`prof_channel_read` 返 0 后再读 `kBiuPerfDrainRetries` (= 4) 次，
   每次仍返 0 才退。捕获飞行中 chunk。**注意**：plan 原版讨论了
   `prof_channel_poll`，但真实 driver 签名是
   `prof_channel_poll(struct prof_poll_info*, int num, int timeout)`——
   全局可读 channel 轮询、timeout 以**秒**计，放在 per-marker drain 热路
   径会阻塞 collector。改为有界重读更轻。
3. **【调试 / 可观测】sentinel region_id**：kernel 末尾打一条已知 id（如
   `kRegionFinSentinel = 4095`），collector drain 结束后检查最后一条 stamp
   是否就是 sentinel。仅作 debug 模式开关使用，正常 run 不开。开了之后能
   量化"飞行 chunk 漏抓率"，发现兜底机制失效时及时告警。

### 4.3 用户 kernel 写法

当前推荐沿用现有 kernel 的写法（直接调 bisheng 内建），见
[`kernel_add_scalar.cpp`](../../../../examples/a5/tensormap_and_ringbuffer/vector_example/kernels/aiv/kernel_add_scalar.cpp)：

```cpp
asm volatile("bar.MTE2");
bisheng::cce::mark_stamp<PIPE_MTE2, 11>();

TLOAD(srcTile, srcGlobal);

asm volatile("bar.MTE2");
bisheng::cce::mark_stamp<PIPE_MTE2, 111>();

set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

pipe_barrier(PIPE_ALL);
bisheng::cce::mark_stamp<PIPE_V, 12>();

TADDS(dstTile, srcTile, scalar);

pipe_barrier(PIPE_ALL);
bisheng::cce::mark_stamp<PIPE_V, 122>();

// ... 末尾建议加 bar.all 触发 trace flush（§4.2 第 1 层兜底）
asm volatile("bar.all");
```

要点：

- **当前验证用 `bisheng::cce::mark_stamp<pipe, id>()` 内建直调**——examples
  里的 kernel 已经在用，落地实测就以这种写法为准。
- region_id 范围 0..4095，由 kernel 作者自由编号，**不强加项目级命名空间**
  （没有 `core_stamp_regions.h` 之类的集中头）。
- MTE 系列自带 `pipe_barrier`（原理文档 §4 的 erratum 规避，由 bisheng
  编译期自动注入）。V/M 上想要 "pipe 完成时刻"语义，显式
  `pipe_barrier(PIPE_V)` 后再 stamp。
- **kernel 末尾建议插 `asm volatile("bar.all")`**（= `pipe_barrier(PIPE_ALL)`），
  让 trace bus 的飞行 chunk 在 FIN 之前落地，规避 §4.2 的硬件假设。

**未来可切换到 `AscendC::MarkStamp` 公开包装**（不在本期实施范围）：

CANN 提供公开包装 `AscendC::MarkStamp<pipe, idx>()`（声明在
`asc/include/basic_api/kernel_prof_trace_intf.h`）。其相对 `bisheng::cce::mark_stamp`
只多做了 3 件**编译期**的事，运行期完全等价（同样一条 `__dfx_region` 指令、
同样的 MTE 自动 barrier）：

| 收益 | 直接用 `bisheng::cce::*` 的对应代价 |
| ---- | ---------------------------------- |
| `ASCENDC_CPU_DEBUG` 自动空实现 | CPU 调试 binary 触发"未支持"，需手动 `#ifdef` 包 |
| `__NPU_ARCH__ == 3510` 编译期门控 | 跨 arch 编译错配只在运行/链接期暴露 |
| 公开 namespace 稳定（`AscendC::`） | `bisheng::cce::*` 是编译器内部 namespace，CANN 升级可能动签名 |

对 simpler A5 项目，前两条用不到（不走 CPU debug、只编 dav-c310），剩"API
稳定性"一条作为升级抗性。CANN 9.1 → 9.2 等版本如果改 `bisheng::cce::mark_stamp`
签名时，再统一切到 `AscendC::MarkStamp` 即可；**现存 kernel 不强制回改，
新写的 kernel 用哪种都可**。

### 4.4 GM 数据结构

```cpp
// include/common/l0_perf_profiling.h
struct L0PerfRecord {
    uint64_t cycle;       // 绝对 cycle（解码 START_STAMP 后重建）
    uint64_t task_id;     // 由 marker 携带
    uint16_t region_id;   // 12 bit 有效
    uint8_t  pipe;        // ctrl_type 0..6
    uint8_t  sub_core_id; // (group * 3 + sub_offset)，0..17
    uint32_t reserved;    // 凑齐 24B 对齐
} __attribute__((aligned(8)));

// AICPU → host 信号（新增，C' 方案核心）
struct L0TaskFinMarker {
    uint64_t task_id;
    uint32_t group;       // 0..5；UINT32_MAX 表示跳过（核不在 biu_perf 覆盖内）
    uint32_t reserved;
} __attribute__((aligned(8)));

// L0PerfModule 的 ReadyEntry payload 就是 L0TaskFinMarker。
// 没有 L0PerfBuffer 轮换、L0PerfFreeQueue、L0PerfBufferState ——
// marker payload 是 16 字节，直接放在 ReadyQueue entry 里。
struct L0PerfReadyQueueEntry {
    L0TaskFinMarker marker;
    uint32_t reserved;     // 保持 padding 对齐 / 给框架元数据预留
};
```

**和 L2 / PMU 的对仗关系**（C' 方案下进一步收敛——L0 不需要 buffer 池）：

| 维度 | L2 | PMU | **L0**（C'） |
| ---- | -- | --- | ----------- |
| AICore-side 写入目标 | per-core stable `L2PerfAicoreRing`（dual-issue 2 槽） | per-core stable `PmuAicoreRing` | 无（HW DFX ring 由 driver 管，HW 是 producer） |
| AICPU producer 写入目标 | rotating `L2PerfBuffer.records[]`（GM） | rotating `PmuBuffer.records[]`（GM） | **无**——AICPU 只 push marker 到 ReadyQueue |
| Buffer state 分槽 | per-AICore | per-AICore | **没有 buffer**——marker 是 entry payload |
| Module kinds | 2（perf + phase） | 1 | 1 |
| Per-thread ReadyEntry payload | buffer pointer + 元数据 | buffer pointer + 元数据 | `L0TaskFinMarker` 直接内联 |
| host consumer 干啥 | resolve buffer + copy from device + collect | 同左 | **resolve marker + 调 driver drain + decode chunks + 写 records** |

**ring vs buffer 的术语澄清**（C' 方案下链路上有两段独立的环形结构）：

| 段 | 谁管 | producer | consumer |
| -- | ---- | -------- | -------- |
| **HW SoC trace ring**（4 MiB/sub-core） | driver | user kernel 的 `mark_stamp` 指令 | **host** collector 通过 `prof_channel_read` |
| **GM ReadyQueue**（per-AICPU-thread） | ProfilerBase 框架 | AICPU scheduler 线程 (FIN 后 push marker) | host mgmt → host collector |

第一段是硬件 ring，driver 内部维护。第二段是 GM 上的 SPSC，载荷从原 C 方
案的 `buffer pointer` 改成 `L0TaskFinMarker` 直接内联，sequencing 完全沿
用 `ProfilerBase` + `BufferPoolManager` 框架（详见
[profiling-framework.md](../../../../docs/profiling-framework.md)）。

**`records[]` 由谁写**：host collector。`on_buffer_collected(marker)` 调
driver drain 拿到 raw chunks → decode → 直接 append 到 collector 自己的
`std::vector<L0PerfRecord>`（或框架的 host-side records 容器）。**GM 里不
再有 `L0PerfBuffer.records[]`** ——AICPU 端不写 record。

## 5. 关键接口序列

`channel_id` 与物理 AICore id 全部集中在 `l0_perf_profiling.h` 的常量集
里。**驱动放开更多核时只改这组常量，业务代码不动**：

```cpp
// include/common/l0_perf_profiling.h
constexpr uint32_t kBiuPerfChanBase    = 11;          // driver 当前给的起点
constexpr uint32_t kBiuPerfNumGroups   = 6;           // 当前 driver 覆盖的 AICore 数
constexpr uint32_t kBiuPerfSubPerGroup = 3;           // aic / aiv0 / aiv1
constexpr uint32_t kBiuPerfNumChans    = kBiuPerfNumGroups * kBiuPerfSubPerGroup;  // 18

// 物理 AICore id 不是 0..5（详见 §6 TODO 2）
constexpr uint32_t kBiuPerfPhysAicore[kBiuPerfNumGroups] = {0, 9, 17, 18, 27, 35};

constexpr int kBiuPerfDrainRetries = 4;   // §4.2 第 2 层兜底

inline uint32_t biu_perf_chan_id(uint32_t group, uint32_t sub) {
    return kBiuPerfChanBase + group * kBiuPerfSubPerGroup + sub;
}

inline int l0_perf_phys_to_group(uint32_t phys_id) {
    for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
        if (kBiuPerfPhysAicore[g] == phys_id) return (int)g;
    }
    return -1;
}
```

### 5.1 Host 端 driver 调用序列（C' 方案的核心）

> **2026-05-27 架构对齐更新**：实测发现只调 `prof_drv_start` +
> `prof_channel_read` 这套 consumer 接口**不够** —— driver 必须先被告知
> "当前进程是 instr-profiling consumer"才会把 DFX trace ring 数据 route
> 到 biu_perf channel。完整必要序列见
> [`core_swimlane_principle.md` §7.5](core_swimlane_principle.md#75-实测必需的-host-启动序列2026-05-26-验证--2026-05-27-架构对齐)。
>
> 简言之：在下方的 `prof_drv_start` 18 通道之前，必须先调
> ```
> aclprofInit + aclprofCreateConfig(ACL_PROF_AICORE_METRICS,
>     ACL_AICORE_PIPE_UTILIZATION) + aclprofStart
> rtProfSetProSwitch(PROF_INSTR=0x00800000, START)
> ```
>
> **架构对齐**：这一段 ACL/runtime 层握手 + dlopen + GM 分配 + 18 路 biu_perf
> 启动**全部封装在 `L0PerfCollector::initialize` 内部**（与 L2/PMU/dump
> 同形 —— DeviceRunner 只调 `collector_.initialize()`）。下方旧版代码示例的
> `dlopen / dlsym / prof_drv_start` 块在新实现里是 initialize 的步骤 3-5,
> 真正的调用顺序见 principle.md §7.5。
>
> 运行期 `on_buffer_collected` 直接在 ProfilerBase 的 collector_thread 上
> 调 `prof_channel_read`（msprof shim 实测 driver 不强制 start/read/stop
> 同线程），不再有额外的 `channel_owner` 线程 / `pending_markers_` 队列。
>
> 同时确认：CCU/STARS/AICPU 9 个 prereq channel（147/148/149/151,
> 44/50/52/53, 143）被 `aclprofStart` 内部接管，**不需要也不能**手动
> `prof_drv_start`（会返 EBUSY=30）。原 plan §6 TODO 5 关于"最小启动契约"
> 的结论作废 —— 真正最小契约是上面那段 ACL+rt 握手。

```cpp
// ───── HOST: L0PerfCollector::initialize(dev_id) ─────
// 在 device_runner 启 collector 时跑一次。dev_id 是 aclrtSetDevice 传进
// 来的真实 device id（dev=0..N）。
struct BiuPerfStartUserData {  // 16 B，所有字段 LE u32
    uint32_t hdr_size;         // = 16，自描述长度
    uint32_t biu_mode;         // 0 = perf monitor
    uint32_t sub_core_type;    // 0 = aic, 1 = aiv0, 2 = aiv1
    uint32_t aicore_phys_id;   // 取自 kBiuPerfPhysAicore[]
};

hal_handle_ = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
prof_drv_start_   = (ProfDrvStartFn)  dlsym(hal_handle_, "prof_drv_start");
prof_channel_read_= (ProfChannelReadFn)dlsym(hal_handle_, "prof_channel_read");
prof_stop_        = (ProfStopFn)      dlsym(hal_handle_, "prof_stop");
// fail-fast on resolve failure

for (uint32_t g = 0; g < kBiuPerfNumGroups; ++g) {
    for (uint32_t sub = 0; sub < kBiuPerfSubPerGroup; ++sub) {
        BiuPerfStartUserData ud{16u, 0u, sub, kBiuPerfPhysAicore[g]};
        prof_start_para sp{};
        sp.channel_type   = PROF_TS_TYPE;
        sp.sample_period  = 0;
        sp.real_time      = 1;
        sp.user_data      = &ud;
        sp.user_data_size = sizeof(ud);
        int r = prof_drv_start_(dev_id, biu_perf_chan_id(g, sub), &sp);
        if (r != 0) {
            rollback_started_channels();
            LOG_ERROR("L0PerfCollector: prof_drv_start(chan=%u) -> %d on dev=%u",
                      biu_perf_chan_id(g, sub), r, dev_id);
            return r;     // fail-fast (与 L2/PMU init 失败一致)
        }
        chan_started_[g * kBiuPerfSubPerGroup + sub] = true;
    }
}
// 返 0 后 ProfilerBase::start() 启 mgmt_thread + collector_thread

// ───── HOST: L0PerfCollector::on_buffer_collected(marker) ─────
// 在 collector_thread 上由 ProfilerBase poll_and_collect_loop 调进来。
// `info` 解出来就是一个 L0TaskFinMarker。
void on_buffer_collected(const ReadyBufferInfo &info) {
    L0TaskFinMarker marker = unpack_marker(info);
    if (marker.group == UINT32_MAX) return;   // 核不在覆盖里，AICPU 端按理也不会 push

    char read_buf[kReadBufSize];
    for (uint32_t sub = 0; sub < kBiuPerfSubPerGroup; ++sub) {
        unsigned chan = biu_perf_chan_id(marker.group, sub);
        int sub_core_idx = (int)marker.group * 3 + (int)sub;
        L0DecodeState &ds = decode_state_[sub_core_idx];

        int retries = kBiuPerfDrainRetries;
        for (;;) {
            int bytes = prof_channel_read_(dev_id_, chan, read_buf, sizeof(read_buf));
            if (bytes > 0) {
                l0_decode_chunks(read_buf, bytes, ds, [&](uint16_t region_id,
                                                          uint8_t pipe,
                                                          uint64_t cycle,
                                                          uint16_t /*block_id*/) {
                    records_.push_back(L0PerfRecord{
                        cycle, marker.task_id, region_id, pipe,
                        (uint8_t)sub_core_idx, 0
                    });
                });
                retries = kBiuPerfDrainRetries;   // got data, reset budget
                continue;
            }
            if (bytes < 0 || retries-- <= 0) break;
        }
    }
}

// ───── HOST: L0PerfCollector::finalize() ─────
for (uint32_t c = 0; c < kBiuPerfNumChans; ++c) {
    if (chan_started_[c]) prof_stop_(dev_id_, kBiuPerfChanBase + c);
}
dlclose(hal_handle_);
write_swimlane_json("<output_prefix>/l0_perf_records.json", records_);
```

### 5.2 AICPU 端代码（仅剩 marker push）

```cpp
// src/aicpu/l0_perf_collector_aicpu.cpp
static int s_l0_logical_to_group[PLATFORM_MAX_CORES];

int l0_perf_aicpu_init(int thread_idx, const int *cores_owned,
                       const uint32_t *phys_ids, int core_num) {
    for (int i = 0; i < core_num; ++i) {
        s_l0_logical_to_group[cores_owned[i]] = l0_perf_phys_to_group(phys_ids[i]);
    }
    return 0;     // 永不失败
}

void l0_perf_aicpu_drain_after_fin(int core_id, int thread_idx, uint64_t task_id) {
    int group = s_l0_logical_to_group[core_id];
    if (group < 0) return;
    L0TaskFinMarker m{task_id, (uint32_t)group, 0};
    enqueue_l0_marker_ready(thread_idx, m);   // 现有 ProfilerBase ready queue
}
```

要点：

- **没有任何 `prof_*` / `dlsym` 调用**——AICPU 端不接触 `libascend_hal.so`
- **没有 `device_id` 参数**——driver 调用全在 host，host 用自己的
  `device_id_` 成员变量
- AICPU 端 `kernel.cpp` 集成点保留 `set_l0_swimlane_enabled` /
  `set_platform_l0_perf_base`，但不再需要 `set_l0_perf_device_id`
  （那是 C 方案残留，C' 删）

## 6. 落地前需要确认的 TODO

- **TODO 1**：✅ 已在 **CANN 9.1.T500** 上验证（host + AICPU 双侧 probe + msprof shim 三方对齐）。
- **TODO 2**：✅ 已在 **CANN 9.1.T500** 上验证（当前环境）。
- **TODO 3**：弃置（CLI 不强加 msprof 互斥检查）。
- **TODO 4**：已并入 §4.2 三层兜底，不再单列。
- **TODO 5**：✅ 已在 **CANN 9.1.T500** 上验证最小启动契约——biu_perf
  `prof_drv_start` 不需要 CCU/STARS 链做前置；同步发现 biu_perf 通道存在
  **per-device 健康状态**，落地需要 init 失败兜底（见下文 TODO 5）。

### TODO 1：biu_perf channel_id 的稳定性 — ✅ 已在 9.1.T500 上验证

**结论**：A5 onboard 上 biu_perf 占用固定的 18 个 channel id（11..28），
按 cluster group + sub-core type 排列。整个 runtime 期间 id 不变；
host/device 两侧都看得到（device-side hal stub 完整暴露 consumer 接口）。

**9.1.T500 上的三方对齐**：

1. **host `prof_drv_get_channels(dev)`**（`/tmp/probe_min.cpp`）：返回 35
   channel 列表，11..28 完整在列；多次重复调用 id 不变；name 字段全空
   ——driver 这个 API 不暴露 name。
2. **AICPU `prof_drv_get_channels(0)`**（早期 `kernel.cpp` 内 dlsym 探针）：
   AICPU 侧通过 device-side `libascend_hal.so` stub resolve 到符号，调用
   返回 num=34（与 host 的 35 仅差 1，且 11..28 完整在列）——证明 device
   端 consumer API linkage + channel 列表都能在 AICPU 里正常用。
3. **msprof shim 追踪**（`/tmp/prof_shim.log`）：msprof 进程对 chan 11..28
   18 个 channel 按
   `chan_id = 11 + group * 3 + sub_offset`、`group ∈ [0,5]`、
   `sub_offset ∈ {0,1,2}` 的顺序 `prof_drv_start`，每次 user_data 的
   `aicore_phys_id` 字段恰好取自 `{0,9,17,18,27,35}`——id ↔ sub-core ↔
   物理核的 (id → name → phys) 映射就是这张表里的样子（name 字符串本身
   来自 9.0.0 plog scraping，但 9.1.T500 上 msprof 行为完全对应）：

| channel_id | name                       | AICore (cluster) | sub-core type |
| ---------- | -------------------------- | ---------------- | ------------- |
| 11         | `biu_perf_group0_aic`      | AICore #0        | cube          |
| 12         | `biu_perf_group0_aiv0`     | AICore #0        | vector 0      |
| 13         | `biu_perf_group0_aiv1`     | AICore #0        | vector 1      |
| 14         | `biu_perf_group1_aic`      | AICore #1        | cube          |
| 15         | `biu_perf_group1_aiv0`     | AICore #1        | vector 0      |
| 16         | `biu_perf_group1_aiv1`     | AICore #1        | vector 1      |
| 17         | `biu_perf_group2_aic`      | AICore #2        | cube          |
| 18         | `biu_perf_group2_aiv0`     | AICore #2        | vector 0      |
| 19         | `biu_perf_group2_aiv1`     | AICore #2        | vector 1      |
| 20         | `biu_perf_group3_aic`      | AICore #3        | cube          |
| 21         | `biu_perf_group3_aiv0`     | AICore #3        | vector 0      |
| 22         | `biu_perf_group3_aiv1`     | AICore #3        | vector 1      |
| 23         | `biu_perf_group4_aic`      | AICore #4        | cube          |
| 24         | `biu_perf_group4_aiv0`     | AICore #4        | vector 0      |
| 25         | `biu_perf_group4_aiv1`     | AICore #4        | vector 1      |
| 26         | `biu_perf_group5_aic`      | AICore #5        | cube          |
| 27         | `biu_perf_group5_aiv0`     | AICore #5        | vector 0      |
| 28         | `biu_perf_group5_aiv1`     | AICore #5        | vector 1      |

公式：`channel_id = 11 + aicore_idx * 3 + sub_offset`
（`sub_offset = 0(aic) / 1(aiv0) / 2(aiv1)`）。代码里通过
`biu_perf_chan_id()` 包装（见 §5）。

⚠ **当前 biu_perf 硬件只支持 6 个 AICore**——见 §10 已知硬件限制。

### TODO 2：`prof_start_para.user_data` 的 biu 模式配置结构 — ✅ 已在 9.1.T500 上验证

**结论**：biu_perf channel 的 `user_data` 是固定 **16 字节** 结构，**没有
group_vector 数组**——msprof 是一个 channel 一个 channel 单独 start 的，
每次 user_data 只指向当前 sub-core：

```c
struct BiuPerfStartUserData {  // size = 16, 全部 LE uint32
    uint32_t hdr_size;         // = 16，等于 user_data_size，自描述
    uint32_t biu_mode;         // 0 = perf monitor（plog "biu mode: 0"）
    uint32_t sub_core_type;    // 0 = aic, 1 = aiv0, 2 = aiv1
    uint32_t aicore_phys_id;   // 物理核 id（注意不是 0..5，见下）
};
```

`prof_start_para` 的其它字段：

| 字段 | 值 |
| ---- | -- |
| `channel_type`     | `PROF_TS_TYPE = 0`（**不是** PERIPHERAL） |
| `sample_period`    | 0 |
| `real_time`        | 1 |
| `user_data_size`   | 16 |

**物理 AICore id 不是 0..5**——biu_perf 实际激活的 6 个核物理 id 为
`{0, 9, 17, 18, 27, 35}`。logical group 索引（TODO 1 表里的 AICore #N）→
物理 id 的映射如下：

| group 索引 | aicore_phys_id | channel_id（aic/aiv0/aiv1） |
| ---------- | -------------- | --------------------------- |
| 0          | 0              | 11 / 12 / 13                |
| 1          | 9              | 14 / 15 / 16                |
| 2          | 17 (0x11)      | 17 / 18 / 19                |
| 3          | 18 (0x12)      | 20 / 21 / 22                |
| 4          | 27 (0x1b)      | 23 / 24 / 25                |
| 5          | 35 (0x23)      | 26 / 27 / 28                |

**验证手段**：写 `src/a5/platform/onboard/host/prof_start_shim.cpp` 这个
独立 .so（**不**链进 host_runtime），用 `LD_PRELOAD` 注入到 msprof 进程：

```bash
g++ -O2 -fPIC -shared -Wall -Wextra \
    -o /tmp/libprof_start_shim.so \
    src/a5/platform/onboard/host/prof_start_shim.cpp -ldl

task-submit --device 3 --run "LD_PRELOAD=/tmp/libprof_start_shim.so \
    PROF_SHIM_LOG=/tmp/prof_shim.log \
    msprof --instr-profiling=on \
    python examples/a5/tensormap_and_ringbuffer/vector_example/test_vector_example.py \
           --device 3 --platform a5 --log-level v0"

grep '\[prof_start_shim\]' /tmp/prof_shim.log
```

shim 用 `dlsym(RTLD_NEXT, "prof_drv_start")` 转发到真实 hal 实现，在
转发前把 `device_id` / `channel_id` / `user_data` hexdump 出来。18 个
biu_perf channel 在 hexdump 里呈现非常规则的字段对齐，加上 plog 中
`Begin to start biu perf job, devId:3, channelId:11, biu mode: 0` 等
锚点，结构直接解出。

**Create group vector number ... different from ... 字符串的真相**：
原推测"user_data 里有 group_vector 数组"被证伪——那条 strings 实际是
driver 在比对**已注册的 channel 数**与**期望数**时打的错误日志，与
user_data 字节布局无关。

**风险残留**：结构在公开头里没有，仍依赖 CANN 版本——但 16 字节 + 自
描述 hdr_size 的设计本身就利于版本兼容，后续 CANN 改字段时 hdr_size
会一起改，运行期可检测。落地代码需在头部加版本/字段 assert。

### TODO 3：channel "占有权" 是否冲突 — 弃置

我们既然自己直接通过 `prof_drv_start` / `prof_channel_read` 拿硬件数据，
就**不与 msprof 共用一次 run**。CLI 层**不强加**互斥检查（成本/收益不
划算），文档警示 "`--enable-l0-swimlane` 与 msprof daemon 同时跑时行为
未定义" 即可。

### TODO 4：drain 完整性 — 已并入 §4.2 三层兜底

原"单线程吞吐 / FIN 时 trace 是否 flush 完"问题已经在 §4.2 用 kernel
barrier + runtime poll 兜底 + sentinel 探针三层吸收。**默认开启第 2 层
（poll 兜底）**即可，不再列为阻塞性 TODO。

### TODO 5：biu_perf 启动的最小契约 + per-device 健康状态 — ✅ 已在 9.1.T500 上验证 + AICPU NOT_SUPPORT 已实测

**结论汇总**（按时间线）：

1. **2026-Q1 verification**（针对 host driver）：仅
   `aclInit + aclrtSetDevice + prof_drv_get_channels + prof_drv_start(11, {16,0,0,0})`
   能让 chan 11..28 在健康 device (dev=2) 上返 0；CCU/STARS 链不是
   biu_perf 的前置。biu_perf 启动存在 **per-device 健康状态**：dev=2 上任何
   调用方（含 msprof）都能开，dev=3 上任何调用方都开不出来（返 -1）。
2. **2026-05-25 verification**（针对 AICPU stub）：device-side
   `/usr/lib64/libascend_hal.so` 的 `prof_drv_start` 在 AICPU 上**端到端
   验证失败**——4 张卡 (dev=0/1/2/3) 全部复现，无论 host driver 健康与否：
    - 错的 `device_id` (用 plan §5 原文的 `kLocalDev=0` 而非 host 的 dev id)
      → AICPU stub 内部 hang，scheduler 后续 STALL → host
      `aclrtSynchronizeStreamWithTimeout` 507018
    - 对的 `device_id` (与 host `aclrtSetDevice` 一致) → 立刻返
      **`DRV_ERROR_NOT_SUPPORT` = `0xFFFE` = 65534**（来自
      `/usr/local/Ascend/cann-9.1.T500/aarch64-linux/pkg_inc/driver/ascend_hal_error.h:144`）
      ——driver 主动表示 AICPU 不支持当 biu_perf consumer
    - dladdr 确认 AICPU 拿到的 `prof_drv_start` 实体地址 = symbol 起点，
      不是 trampoline / 弱符号 stub，CANN 部署到 device rootfs 的那份
      libascend_hal.so（38 KB 量级）是 biu_perf consumer "未实现"
3. **架构调整**：plan §4.1 原版 "AICPU 当 consumer"路线**被 driver 拒**。
   方案从 C → C' （host 当 consumer，AICPU 只 push marker），详见顶部
   修订块与重写后的 §3 / §4.1 / §5。

**`/usr/lib64/libascend_hal.so` 与 host driver lib 的关系**：

| 路径 | 大小 | 谁用 | 谁实现 |
| --- | --- | --- | --- |
| `/usr/local/Ascend/driver/lib64/driver/libascend_hal.so` | 3.05 MB | host 进程 | NPU driver package（PCIe userspace driver，包含完整 ioctl 链路） |
| `/usr/lib64/libascend_hal.so`（device rootfs 内） | 38 KB | AICPU 进程 | CANN devlib 部署的 device-side stub（biu_perf consumer 未实现） |

两份**ELF 同架构 aarch64**，**导出同名符号**（都有 `prof_drv_start`），
但**实现独立**——host 那份会跑 `ioctl(/dev/davinci_*, ...)` 走 PCIe 控制
平面，AICPU 那份跑在 NPU 内部、下面没有 PCIe / 没有内核驱动模块。CANN
对 device 侧的 prof consumer API **没真实现**，返 NOT_SUPPORT 是设计意
图——这也解释了为什么 a5/a2a3 其它 profiling 子系统（L2/PMU/dump）都没踩到
这块（它们 AICPU 端不调任何 driver API，纯 GM 指针 + rtMemcpy 回流）。

**host 一侧验证脚本**（保留以备日后回归测）：
`/tmp/l0_probe_min.cpp` ——
`aclInit + aclrtSetDevice(dev) + prof_drv_get_channels + prof_drv_start(11/20)`，
跑 `task-submit --device N --run "/tmp/l0_probe_min {}"` 看返回码。

**AICPU 一侧验证脚本**（嵌在 simpler runtime 内）：见
`src/a5/platform/src/aicpu/l0_perf_collector_aicpu.cpp` 在 C 方案时期
加的 dladdr + call-bracket 诊断 log。C' 方案落地后这些 log 与 dlsym
逻辑随 AICPU 端代码一起删除。

**落地策略**（与 L2 / PMU / dump 子系统对齐）：

- host `L0PerfCollector::initialize()` 调 `prof_drv_start` 失败时 fail-fast
  返错，与 L2/PMU init 失败一致；不退化为"L0 静默关闭"，避免误报"采到完整
  数据"
- biu_perf per-device 健康状态依旧存在；dev=3 这类卡上 host 也会拿到 -1，
  collector init 返错 → 上层报错 → 不进 main run，与原 plan 策略相同

## 7. 与现有 profiling 子系统的关系

| 子系统 | 数据通路 | 触发开关 | 与 L0 perf 关系 |
| ------ | -------- | -------- | -------------- |
| L2 swimlane | AICore 写 GM → AICPU commit → host | `--enable-l2-swimlane` | **独立**通道；L0 perf 复用其 task→core 映射 |
| PMU | AICore 写 GM → AICPU commit → host | `--enable-pmu=N` | 完全独立 |
| Tensor dump | AICore 写 GM → AICPU commit → host | `--enable-dump-tensor` | 完全独立 |
| **L0 perf（C' 方案）** | HW DFX bus → SoC trace ring → **host** `prof_channel_read` (in `on_buffer_collected(marker)`) → 写 host-side L0PerfRecord vector → 落 JSON。AICPU 端只 push `L0TaskFinMarker` 到 GM ReadyQueue | `--enable-l0-swimlane` | 复用 ProfilerBase 框架；与 L2/PMU 同形（host derived collector）；**唯一一个 host 端调 libascend_hal 的 collector** |

四个子系统都继承相同基类 `ProfilerBase<...>`（见
[`profiling-framework.md`](../../../../docs/profiling-framework.md)）。L0 是
首个**host 端调 driver consumer API** 的子系统——L2/PMU/dump 都不接触
`libascend_hal.so`，纯 GM 通信。

## 8. 改动清单

按 [profiling-framework.md §7 "Adding a new collector"](../../../../docs/profiling-framework.md)
的步骤对仗，C' 方案下的最终面貌。具体行数取决于实现时的折衷。

```text
新增（platform 侧）：
  include/common/l0_perf_profiling.h         L0PerfRecord + L0TaskFinMarker + L0PerfReadyQueueEntry
                                              + L0PerfDataHeader (per-thread ready queue 数组)
                                              + kBiuPerf* 常量集 + biu_perf_chan_id() / l0_perf_phys_to_group() helper
                                              + l0_decode_chunks() chunk 解码 (host 用)
  include/host/l0_perf_collector.h           L0PerfModule trait struct
                                              + L0PerfCollector : ProfilerBase<L0PerfCollector, L0PerfModule>
                                              + BiuPerfStartUserData (16B host-only)
                                              + ProfDrvStartFn / ProfChannelReadFn / ProfStopFn 类型
                                              + chan_started_[18] / decode_state_[18] / records_ 状态
  include/aicpu/l0_perf_collector_aicpu.h    extern "C" 极简接口:
                                                set_platform_l0_perf_base / get_platform_l0_perf_base
                                                set_l0_swimlane_enabled / is_l0_swimlane_enabled
                                                l0_perf_aicpu_init(thread_idx, cores_owned, phys_ids, core_num)
                                                l0_perf_aicpu_drain_after_fin(core_id, thread_idx, task_id)
                                                  ↑ 函数名沿用旧版便于 scheduler 集成点不变；
                                                    现在语义是 "enqueue L0TaskFinMarker"
  src/host/l0_perf_collector.cpp             initialize: dlopen + dlsym + 18 prof_drv_start (fail-fast)
                                              on_buffer_collected: 3 channel drain + decode + append records
                                              finalize: 18 prof_stop + dlclose + write JSON
                                              ~250 行
  src/aicpu/l0_perf_collector_aicpu.cpp      建 logical→group 表 + push marker
                                              ~50 行（比 C 方案的 ~400 行少 ~80%）

最小扩展：
  include/common/platform_config.h           +PROFILING_FLAG_L0_SWIMLANE (bit3，与 dump/l2/pmu 同形)
                                              +PLATFORM_L0_PERF_READYQUEUE_SIZE (marker queue capacity)
                                              +PLATFORM_L0_PERF_TIMEOUT_SECONDS (kIdleTimeoutSec)
  include/common/kernel_args.h               +KernelArgs.l0_perf_data_base (uint64_t)
                                              （enable_profiling_flag bit3 即 L0_SWIMLANE，
                                                bitmask 已有的复用；device_id 字段在 C' 下不需要，可删）
  onboard/aicpu/kernel.cpp                   ~line 116 处加 set_l0_swimlane_enabled(...) +
                                              set_platform_l0_perf_base(...)（不再需要 set_l0_perf_device_id）
  tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp:
                                              每个线程拿到 cores_owned 之后:
                                                if (is_l0_swimlane_enabled())
                                                    l0_perf_aicpu_init(thread_idx, cores_owned,
                                                                       phys_ids, core_num);
                                                    // 返 0 永真，不必 if (rc != 0) return -1
                                              shutdown flush 块**删除** l0_perf_aicpu_flush_buffers 调用
                                              （AICPU 端没有 buffer 要 flush）
  tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp:
                                              L2 complete_record 块后:
                                                if (is_l0_swimlane_enabled())
                                                    l0_perf_aicpu_drain_after_fin(core_id, thread_idx, task_id);
                                              （集成点不变，语义已经从 drain 变 push marker）
  host/device_runner.cpp                     +L0PerfCollector 实例 + initialize(device_id_)/start(tf)/stop/finalize lifecycle
                                              + 把 collector.get_l0_perf_setup_device_ptr() 写到
                                                KernelArgs.l0_perf_data_base
                                              + 设 enable_profiling_flag bit3
                                              **删除** kernel_args_.args.device_id = device_id_ 那行
                                                （C 方案残留，C' 不需要透 device_id 给 AICPU）
  scene_test.py / call_config.py             --enable-l0-swimlane CLI 旗（与 --enable-l2-swimlane 同形）
  simpler_setup/tools/swimlane_converter.py  读 l0_perf_records.json，per-task 下方挂 per-pipe sub-lane

不动:
  ProfilerBase / BufferPoolManager / ProfilerAlgorithms 框架本身
  L2 / PMU / TensorDump 任何代码
  Kernel 函数签名（用户 kernel 自由插 mark_stamp，kernel.cpp:127 那条 set_l0_perf_device_id 删）
  AICPU 线程模型（不新增常驻线程，沿用 ProfilerBase 自带的 mgmt + collector 两线程）
  msprof / libprofimpl 任何接口
  Handshake 结构

从 C 方案删除（彻底）:
  AICPU 端 resolve_prof_fns + g_prof_drv_start/g_prof_channel_read/g_prof_stop dlsym
  AICPU 端 BiuPerfStartUserData / L0ProfStartPara 本地 mirror 结构
  AICPU 端 chunk 解码（搬到 host）
  AICPU 端 L0PerfBuffer 写入路径（l0_append_record / l0_switch_buffer / l0_pop_initial_buffer）
  AICPU 端 dladdr + call-bracket 诊断 log（C 方案排障时加的）
  KernelArgs.device_id 字段 + 配套 set_l0_perf_device_id setter
  l0_perf_aicpu_flush_buffers 接口（AICPU 端无 buffer 可 flush）
```

## 9. 不在本文范围

- sim 平台支持（无真实 SoC trace buffer，硬件路径在 sim 上不可达；如果
  sim 要看核内泳道得另走软件 stamp 路径，与本方案完全独立）
- MindStudio / Perfetto viewer 直接消费——本方案产物只走项目自有
  `swimlane_converter.py`
- AICPU 上 prof channel 多实例 / 多 device 编排——单 device 走通后再加
- 切换到 `AscendC::MarkStamp` 公开包装（§4.3 末尾记录了迁移条件，等 CANN
  升级真的动了 `bisheng::cce::mark_stamp` 签名时再做）

---

## 10. 已知硬件限制：当前覆盖 6 个 AICore

A5 一颗 SoC 物理上有 **36 个 AICore**（每核 = 1 cube + 2 vector，又叫
cluster / blockdim）。但 biu_perf 硬件采集**当前 CANN 版本只覆盖 6 个
AICore**（即 18 个 sub-core，对应 channel_id 11..28，见 §6 TODO 1 表）。
这 6 个核的物理 id 为 `{0, 9, 17, 18, 27, 35}`（见 §6 TODO 2），不是
逻辑连续的 0..5——这是**硬件 + driver 的能力上限**，软件层面无法绕开。

### 10.1 对作业的实际影响

| block_dim | 覆盖情况 | 影响 |
| --------- | -------- | ---- |
| ≤ 6 而且全部落在 phys {0,9,17,18,27,35} | ✅ 所有 task 都在被采核上，stamp 完整 | 完整核内泳道 |
| 否则 | ⚠ 只有命中 phys {0,9,17,18,27,35} 的 task 有 stamp | 部分核内泳道；超出核的 task 在 viewer 里只显示 L2 swimlane 横条，没有 per-pipe 子 lane |

scheduler 的 task→AICore 映射策略（哪个 task 跑哪个核）会**间接决定**用户
能看到多少 stamp 数据。当前不需要也不应该改调度策略迎合 biu_perf——只是
作为已知限制写明。

### 10.2 扩核路线 = 改常量

驱动放开更多 phys_id 进 biu_perf 时，simpler 侧的工作量是**改 `l0_perf_profiling.h`
里的一组常量**：

```cpp
constexpr uint32_t kBiuPerfNumGroups = 12;   // ← 6 → 12
constexpr uint32_t kBiuPerfPhysAicore[kBiuPerfNumGroups] = {
    0, 9, 17, 18, 27, 35,                    // 原 6 个
    /* 新增 phys_id，由 driver 公布 */
};
```

业务代码（scheduler drain 循环、host collector、`KernelArgs` 字段、CLI
warning 阈值）**全部不动**。

阶段 3（36 AICore 全覆盖）需要 CANN 版本进一步演进，不在 simpler 项目范围。

### 10.3 给用户的 CLI 信号

当 `--enable-l0-swimlane` 启用且本次 run 的 task 可能落到 phys
`{0,9,17,18,27,35}` 之外时，host 应在 CLI 输出一行 warning：

```text
[l0-swimlane] biu_perf only covers AICore phys {0,9,17,18,27,35};
              tasks scheduled on other cores will have no per-pipe lane.
```

让用户清楚知道当前的覆盖范围，避免把"没数据"误判为 bug。

---

## 11. 实施细化（CRTP 框架下的具体形态）

按 [profiling-framework.md](../../../../docs/profiling-framework.md) §3.3 /
§3.4 / §7 的契约写。下面给最关键的几段轮廓——所有写法严格对仗 PMU（单
kind 是最接近的现存参考），与 [pmu_collector.h](../include/host/pmu_collector.h)
diff 起来读最容易上手。

### 11.1 `L0PerfModule` trait（layout 声明）

```cpp
// include/common/l0_perf_profiling.h
struct L0TaskFinMarker {
    uint64_t task_id;
    uint32_t group;       // 0..5；UINT32_MAX 表示跳过（核不在 biu_perf 覆盖内）
    uint32_t reserved;
} __attribute__((aligned(8)));

struct L0PerfReadyQueueEntry {     // ReadyQueue 里直接内联 marker，没有 buffer 指针
    L0TaskFinMarker marker;
};

struct L0PerfDataHeader {          // 沿用 PMU 的 per-thread queue 数组形态
    uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];
    uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];
    L0PerfReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_L0_PERF_READYQUEUE_SIZE];
};
```

```cpp
// include/host/l0_perf_collector.h
struct L0PerfReadyBufferInfo {     // 与 PmuReadyBufferInfo 同形，但 payload 是 marker
    L0TaskFinMarker marker;
    uint32_t thread_index;
};

struct L0PerfModule {
    using DataHeader      = L0PerfDataHeader;
    using ReadyEntry      = L0PerfReadyQueueEntry;
    using ReadyBufferInfo = L0PerfReadyBufferInfo;
    // 没有 FreeQueue —— marker 是 entry 内联，不需要 buffer 池

    static constexpr int      kBufferKinds      = 1;
    static constexpr uint32_t kReadyQueueSize   = PLATFORM_L0_PERF_READYQUEUE_SIZE;
    static constexpr const char *kSubsystemName = "L0PerfModule";

    static DataHeader *header_from_shm(void *shm) { return get_l0_perf_header(shm); }

    static std::optional<profiling_common::EntrySite<L0PerfModule>>
    resolve_entry(void *shm, DataHeader *header, int q, const ReadyEntry &entry) {
        profiling_common::EntrySite<L0PerfModule> site;
        site.kind = 0;
        site.free_queue = nullptr;     // marker 不需要 buffer 池
        site.buffer_size = 0;
        site.info.marker = entry.marker;
        site.info.thread_index = static_cast<uint32_t>(q);
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *, DataHeader *, Cb &&) {
        // no-op：L0 不分 buffer instance
    }
};
```

ProfilerBase 的 `process_entry` 在 marker payload 上跑会跳过
`copy_buffer_from_device`（因为 `site.buffer_size == 0`），但仍把
`ReadyBufferInfo`（含 marker）push 到 host `manager_.ready_queue`，
collector_thread 在 `on_buffer_collected` 里拿到。

### 11.2 `L0PerfCollector`（Derived）

```cpp
class L0PerfCollector
    : public profiling_common::ProfilerBase<L0PerfCollector, L0PerfModule> {
public:
    static constexpr int kIdleTimeoutSec = PLATFORM_L0_PERF_TIMEOUT_SECONDS;
    static constexpr const char *kSubsystemName = "L0Perf";

    int initialize(int device_id, const std::string &output_prefix);
    void on_buffer_collected(const ReadyBufferInfo &info);
    int  export_swimlane_json();         // <prefix>/l0_perf_records.json
    int  finalize();

    void *get_l0_perf_setup_device_ptr() const { return perf_shared_mem_dev_; }

private:
    int  dev_id_{-1};
    void *hal_handle_{nullptr};
    ProfDrvStartFn    prof_drv_start_{nullptr};
    ProfChannelReadFn prof_channel_read_{nullptr};
    ProfStopFn        prof_stop_{nullptr};

    bool chan_started_[kBiuPerfNumChans] = {false};
    L0DecodeState decode_state_[kBiuPerfNumChans];
    std::vector<L0PerfRecord> records_;
    char read_buf_[8192];

    void *perf_shared_mem_dev_{nullptr};
    std::string output_prefix_;
};
```

`initialize()` 流程：
1. `hal_handle_ = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL)`
2. `dlsym` 三个 `prof_*` 符号；任一为 NULL 走 fail-fast
3. for chan in 11..28: `prof_drv_start(dev_id_, chan, &ud)`，失败回滚 + 返错
4. 调 base 类的 `init_shared_mem()` 等，准备 GM ReadyQueue

`on_buffer_collected(info)` 流程：
1. `marker = info.marker`；`group = marker.group`
2. 对 group 的 3 个 sub-core：
   - `prof_channel_read(dev_id_, chan, read_buf_, sizeof(read_buf_))`
   - `bytes > 0` → `l0_decode_chunks(...)` callback 把 `(cycle, pipe,
     region_id)` 加上 `marker.task_id` + `sub_core_id` push 到 `records_`
   - `bytes == 0` → 有界 retry `kBiuPerfDrainRetries` 次
3. 退出后什么都不做，框架会 `notify_copy_done`

`finalize()` 流程：
1. for chan in 11..28: `prof_stop(dev_id_, chan)`
2. `dlclose(hal_handle_)`
3. `export_swimlane_json()` 写 `<prefix>/l0_perf_records.json`

### 11.3 AICPU 端 extern "C" 接口（对仗 `l2_perf_collector_aicpu.h`）

```cpp
// include/aicpu/l0_perf_collector_aicpu.h
extern "C" void     set_platform_l0_perf_base(uint64_t addr);
extern "C" uint64_t get_platform_l0_perf_base();
extern "C" void     set_l0_swimlane_enabled(bool enable);
extern "C" bool     is_l0_swimlane_enabled();

// 每个 scheduler 线程拿到 cores_owned 之后调一次。
// 内部仅做：把 logical core id → biu_perf group (0..5 / -1) 写入静态表。
// **永不失败**——不调 driver，没有可能出错的资源申请。
int  l0_perf_aicpu_init(int thread_idx, const int *cores_owned,
                        const uint32_t *phys_ids, int core_num);

// task FIN 后调（scheduler_completion.cpp 紧跟 l2_perf complete_record）。
// 函数名 "drain_after_fin" 是历史名（C 方案时期是真 drain），现在语义已经
// 变成 "enqueue L0TaskFinMarker"——保留旧名让 scheduler 集成点不变。
void l0_perf_aicpu_drain_after_fin(int core_id, int thread_idx, uint64_t task_id);
```

**对比 C 方案删除的接口**：

```cpp
// 不再需要：
void set_l0_perf_device_id(uint32_t device_id);      // device id 在 host
void l0_perf_aicpu_flush_buffers(int thread_idx,     // AICPU 端无 buffer 可 flush
                                  const int *cores_owned, int core_num);
```

### 11.4 集成点（3 处 diff，比 C 方案少 1 处）

| 文件 | 锚点 | 新增 |
| --- | --- | --- |
| [`onboard/aicpu/kernel.cpp`](../onboard/aicpu/kernel.cpp) `:~116` | `set_pmu_enabled(...)` 之后 | `set_l0_swimlane_enabled(GET_PROFILING_FLAG(..., PROFILING_FLAG_L0_SWIMLANE))` + `set_platform_l0_perf_base(k_args->l0_perf_data_base)` |
| [`scheduler_dispatch.cpp`](../../../runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp) `:~382`（每线程 `l2_perf.reset()` 之后） | thread 拿到 `cores_owned[]` 之后 | `if (is_l0_swimlane_enabled()) l0_perf_aicpu_init(thread_idx, cores_owned, phys_ids, core_num);`（永远返 0，不需要 fail-fast 分支） |
| [`scheduler_completion.cpp`](../../../runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp) `:~195` | `l2_perf_aicpu_complete_record(...)` 块之后 | `if (is_l0_swimlane_enabled()) l0_perf_aicpu_drain_after_fin(core_id, thread_idx, slot_state.task->task_id.raw);` |

**从 C 方案删除**：`scheduler_dispatch.cpp` 的 flush 块里调
`l0_perf_aicpu_flush_buffers` 那一行直接删——AICPU 端没有 buffer 要 flush。

host 侧：[`device_runner.cpp`](../onboard/host/device_runner.cpp) 加
`L0PerfCollector` 实例的 `initialize(device_id_, output_prefix) / start(tf) /
stop / finalize` 生命周期，把 `collector.get_l0_perf_setup_device_ptr()`
写到 `KernelArgs::l0_perf_data_base`，并在 `KernelArgs::enable_profiling_flag`
里置 `PROFILING_FLAG_L0_SWIMLANE` bit。**删除** `KernelArgs::device_id`
那行 + 配套字段（C 方案残留）。

**L0 单开 = 仅落 raw JSON，不出 merged_swimlane**：L0 的 per-pipe stamp 在
viewer 里按 sub_core × pipe 分 lane（converter 里的 pid=5 视图）——若同核
跑多个 task，所有 task 的 stamp 会串在同一条 pipe lane 上，没有 L2 task
横条做时间锚就读不出"哪段属于哪个 task"。所以
`scene_test.run_class_cases` 只在 `--enable-l2-swimlane` 开启时调
`swimlane_converter.py`；仅给 `--enable-l0-swimlane` 时**只写出
`l0_perf_records.json`**（其中已带 `task_id` 字段供后续工具按任务分析），
不生成 `merged_swimlane.json`。要看合并 timeline 请同时开 L2。

### 11.5 落地切片（按依赖排序）

> **落地状态（2026-05-25）**：
> - Slice 0-4（C 方案，AICPU 当 consumer）：代码合入 a5/l0swimlane 分支，
>   chunk 解码器单测 `tests/ut/cpp/a5/test_l0_decode.cpp` 通过；端到端**实测
>   不可行**——AICPU `prof_drv_start` 在所有 device 上返
>   `DRV_ERROR_NOT_SUPPORT` (§6 TODO 5)。
> - **Slice 5（C' 架构调整，进行中）**：consumer 上移到 host，AICPU 端只 push
>   marker。详见下文 Slice 5 计划。
>
> 落地相对本文的几处 API 修正（按真实 `ascend_hal_base.h` / msprof 解析侧）：
> 1. **§4.2 第 2 层兜底不用 `prof_channel_poll`**。真实签名是
>    `prof_channel_poll(struct prof_poll_info*, int num, int timeout)`——
>    全局可读 channel 轮询、timeout 以**秒**计，放在 per-marker drain 热路径
>    会阻塞 collector。改为 `prof_channel_read` 返 0 后做有界**重读**重试
>    （`kBiuPerfDrainRetries`，见 `l0_perf_profiling.h`）；kernel `bar.all`
>    （第 1 层）仍是主要 flush 保证。
> 2. **`l0_perf_aicpu_drain_after_fin` 增加 `thread_idx` 形参**（本文 §11.3
>    已更新）——ready-queue push 需要 thread 归属，`scheduler_completion.cpp`
>    已持有 `thread_idx`，与 PMU 一致。
> 3. **chunk 字段位置**以 msprof 解析侧为准（ctrl_type 在高 4 bit、region 在
>    bit 27..16、sys_cnt 在低 16 bit），已在 principle 文档 §6 更正。
> 4. **`l0_perf_aicpu_init` 增加 `phys_ids` 形参**（本文 §11.3 已更新）——
>    logical→group 表由各线程从 `physical_core_ids_[cores_owned[i]]` 现取现传，
>    避免 AICPU 侧重复维护一份全局 physical id 视图。
> 5. **L0 单开只落 raw JSON**：L0 sub_core × pipe 视图在同核多 task 时会把
>    多个 task 的 stamp 串到同一条 pipe lane 上，不靠 L2 task 横条做时间
>    锚就读不出归属。`scene_test.run_class_cases` 只在 L2 开启时跑
>    `swimlane_converter.py`；仅给 `--enable-l0-swimlane` 时只写出带
>    `task_id` 字段的 `l0_perf_records.json`，不生成 `merged_swimlane.json`
>    （详见 §11.4 末）。

**🟢 Slice 0–4 (C 方案，已弃)**

仍存在于 git 历史，代码大部分由 Slice 5 重写或删除。要点：

- Slice 0 / TODO 1 / TODO 2 / TODO 5 验证记录在 §6 仍然有效（biu_perf
  channel id 11..28、user_data 16 字节结构、host driver 行为）
- chunk 解码器 `l0_decode_chunks` + 单测 `test_l0_decode.cpp` 保留
  （Slice 5 host collector 复用）
- `L0PerfRecord` 结构定义保留（最终落盘数据格式不变）

**🔄 Slice 5（进行中）—— C → C' 架构调整：consumer 上移到 host**

按依赖顺序：

1. **doc 更新**：本计划文档全面修订（顶部修订块 + §1-§5 + §6 TODO 5 + §7 /
   §8 / §11）。**优先做**——其它实现按新 plan 跑
2. **common header 调整**：`include/common/l0_perf_profiling.h`
   - 新增 `L0TaskFinMarker` / `L0PerfReadyQueueEntry`（marker 内联）
   - 简化 `L0PerfDataHeader`（删 buffer 池相关字段）
   - **删** `L0PerfBuffer` / `L0PerfBufferState` / `L0PerfFreeQueue`
   - **删** `BiuPerfStartUserData` / `L0ProfStartPara`（搬 host）
   - `kBiuPerf*` 常量集保留，`l0_decode_chunks` 保留
3. **AICPU 端简化**：`src/aicpu/l0_perf_collector_aicpu.cpp` 从 ~400 行减到
   ~50 行，仅剩 `set_*` setters + `l0_perf_aicpu_init` 建映射表 +
   `l0_perf_aicpu_drain_after_fin` push marker。
   - 删 `resolve_prof_fns` / `g_prof_*` / dlsym 全套
   - 删 `BiuPerfStartUserData` / `L0ProfStartPara` 本地 mirror
   - 删 chunk 解码（搬 host）
   - 删 `L0PerfBuffer` 写入、`l0_switch_buffer`、`l0_pop_initial_buffer`
   - 删 dladdr + call-bracket 诊断 log（C 方案排障时加的）
   - 删 `set_l0_perf_device_id` 和 `g_l0_perf_device_id`（C 方案排障时加的）
   - 删 `l0_perf_aicpu_flush_buffers` 接口
4. **host 端扩展**：`include/host/l0_perf_collector.h` + `src/host/l0_perf_collector.cpp`
   - 加 `dlopen("libascend_hal.so", RTLD_NOW|RTLD_LOCAL)` + dlsym
   - 加 `BiuPerfStartUserData` (16B)
   - `initialize()` 跑 18 个 `prof_drv_start`（fail-fast + rollback）
   - `on_buffer_collected(marker)` 跑 drain + decode + append
   - `finalize()` 跑 18 个 `prof_stop` + `dlclose` + 写 JSON
5. **scheduler 集成点修剪**：
   - `scheduler_dispatch.cpp` init 块：`l0_perf_aicpu_init` 调用永远返 0，
     删 `if (rc != 0) return -1` 分支
   - `scheduler_dispatch.cpp` flush 块：删 `l0_perf_aicpu_flush_buffers` 调用
   - `scheduler_completion.cpp`：保留 `l0_perf_aicpu_drain_after_fin` 调用
     不变（语义已经变 push marker）
6. **kernel.cpp**：删 `set_l0_perf_device_id` 调用
7. **device_runner.cpp**：删 `kernel_args_.args.device_id = device_id_`；
   `L0PerfCollector::initialize` 改为传 `device_id_`
8. **kernel_args.h**：删 `device_id` 字段
9. **端到端验证**：dev=2 跑 `kernel_add_scalar.cpp` example，看
   `l0_perf_records.json` 有合理 records；与 L2 swimlane 合并出
   `merged_swimlane.json`

### 11.6 几个落地时容易踩的坑（提前 flag）

**C' 方案下：**

1. **AICPU 端没有 `prof_*` 调用** —— code review 如果看到 AICPU 端出现
   `dlsym`/`prof_drv_start`/`prof_channel_read` 就是回到 C 方案的路了，
   driver 会返 NOT_SUPPORT。所有 driver consumer 调用必须在 host
   `L0PerfCollector` 上。
2. **`core_id → biu_perf_group_idx` 映射** —— scheduler 里 `core_id` 是
   logical id，需要查 `physical_core_ids_[core_id]` 拿到 phys id，再查 phys
   id 是否 ∈ `{0,9,17,18,27,35}`。封装成
   `int l0_perf_phys_to_group(int phys_id)` 返回 -1 / 0..5。这个映射在
   `l0_perf_aicpu_init` 里建表，`l0_perf_aicpu_drain_after_fin` 走表查。
3. **chunk 解码搬到 host 后** —— `l0_decode_chunks` + 单测
   `test_l0_decode.cpp` 不变；测试链接关系从 AICPU lib 切到 host lib。
4. **`PROFILING_FLAG_L0_SWIMLANE` bit 位** —— `enable_profiling_flag`
   bit0/1/2 已被 dump/l2/pmu 占；L0 用 bit3。落地时检查 a2a3 的同位是否
   也空着，避免日后两 arch flag 漂移。
5. **`prof_drv_start` 失败时**不要**降级"静默关闭 L0"** —— 与 L2 / PMU
   保持一致：init 失败 = 上层报错（详见 §6 TODO 5）。biu_perf 有
   per-device 健康状态：dev 处于异常时 host 也开不出（返 -1），collector
   `initialize()` fail-fast 返错；`rollback_started_channels()` 把当前已经
   start 成功的 channel 全部 `prof_stop` 干净再返错。
6. **不要把 `prof_drv_start` 拆到多线程** —— L0PerfCollector 自己用单线程
   做 start + read + stop，避开 plan 原本担心的 "channel 按调用线程归属"
   问题。如果将来想 sharding 给多个 worker，先做 host 双线程 probe 验证。
7. **延迟放大风险** —— mgmt 拉 marker → collector pop → driver drain 的端
   到端延迟通常 ~50μs（mgmt 10μs tick + collector 阻塞唤醒）。若某条 kernel
   特别短（< 50μs）且未插 `bar.all`，drain marker_N 时可能拉到 task N+1 的
   chunk。kernel 末尾插 `bar.all` 是当前**强烈推荐**写法，不强制——后续若
   实际遇到再考虑加 sentinel region_id 4095 探针检测错位（§4.2 第 3 层）。

**已 obsolete（C 方案残留，C' 下不再适用）：**

- ~~L0 init 的"每线程"vs L2 init 的"thread 0"~~ —— C' 下 AICPU 不调
  `prof_drv_start`，"channel 按调用线程归属"的约束转移到 host，host 用
  单线程绕开
- ~~drain 实测 chunk 解码（AICPU 端）~~ —— 解码搬到 host，AICPU 端无解码
  代码
