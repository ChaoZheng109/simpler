# A5 核内泳道图（L0 Swimlane）原理

本文只讲清楚 "在 A5（dav-c310）上，`mark_stamp` 数据是怎么产生、怎么落盘、最终
在 viewer 里画成核内泳道图的"——基于当前机器上的 CANN 9.1.T500
（`/usr/local/Ascend/cann-9.1.T500`，T 系列预发布，版本号即目录名；
原始撰写基线为 CANN 9.0.0 / `version=26.0.rc1`，结构层结论与头文件路径
在 9.1.T500 上原位保留）。**纯原理参考**，不包含
项目侧的实施方案（实施方案见 [core_swimlane_plan.md](core_swimlane_plan.md)），
也不涉及 sim 平台。

---

## 1. 核内泳道图是什么

**核内泳道（Intra-Core Swim Lane / L0 Swim Lane）**指的是把**单个 AICore 内部**
按流水（pipe）拆成的子时间轴：在一个 task 跑的过程中，V / M / MTE1 / MTE2 /
MTE3 / FIX 各 pipe 分别在哪些时间段被使用。

它和 simpler 现有的 `--enable-l2-swimlane`（per-task `start_time`/`end_time` +
AICPU 调度阶段）是两个层次：

| 层次 | 粒度 | 数据来源 |
| ---- | ---- | -------- |
| L2 swimlane（已有） | 每个 task 一条横条 | AICore 写 `get_sys_cnt` 到 GM；AICPU commit 到 host |
| **L0 swimlane（本文）** | task 内部，每 pipe 一条子横条 | AICore `mark_stamp` → SoC DFX 硬件直采 |

核内泳道回答的是 "task 内部时间花在哪个 pipe / 哪个阶段"，
L2 swimlane 回答的是 "哪些 task 各自占了哪段墙钟"。

---

## 2. AICore 流水线模型（理解后面所有"精度"讨论的前提）

AICore 是**多发射异构流水**：

```text
                ┌─────────────────────┐
  指令流 ─────►│  Scalar 派发器       │  顺序执行 scalar 指令：
                └──────┬──────────────┘   get_sys_cnt / 标量算术 / 控制流 / 把
                       │                  vec/mte 指令塞入对应 pipe 队列
       ┌───────┬───────┼───────┬────────┬───────┐
       ▼       ▼       ▼       ▼        ▼       ▼
   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
   │PIPE_V││PIPE_M││ MTE1 ││ MTE2 ││ MTE3 ││ FIX  │  各 pipe 独立队列，异步执行
   └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
```

关键性质：

- **Scalar 派发顺序**：scalar 流水按程序顺序一条接一条执行；`get_sys_cnt()`
  也是 scalar 指令，立即返回当前 scalar 周期。
- **Pipe 之间异步**：scalar 把一条 V 指令塞进 V queue 后**立即继续**派发后面
  的指令，**不等** V pipe 把它跑完。MTE 同理。
- **同步靠显式 barrier**：`pipe_barrier(p)` 是 scalar 流水上的等待指令，让
  scalar 暂停直到 pipe `p` 的所有已发射指令完成。

这条性质决定了：**"现在 V pipe 跑到哪里了" ≠ "现在 scalar 跑到哪里了"**。
mark_stamp 的精度边界完全来自这条。

---

## 3. `AscendC::MarkStamp` 在 CANN 9.x 上的展开链

dav-c310（A5）在 CANN 9.x 中的 `__NPU_ARCH__` 值是 **3510**（旧 CANN 8.x 中
为 3101，CANN 9.0 起改成 3510；本机 9.1.T500 头文件保留同一值）。`AscendC::MarkStamp<>` 仅在该宏匹配时才存在。

### 3.1 调用链

用户侧用法（公共 API）：

```cpp
#include <basic_api/kernel_prof_trace_intf.h>

AscendC::MarkStamp<pipe_t::PIPE_V, 7>();        // 模板版（id 编译期常量）
AscendC::MarkStamp<pipe_t::PIPE_V>(dynamic_id); // 运行期 id 版
```

从公共 API 到硬件指令，CANN 9.x 的展开链如下（9.1.T500 头文件实际确认）：

```text
AscendC::MarkStamp<pipe, idx>()             [interface: asc/include/basic_api/kernel_prof_trace_intf.h]
    │
    ▼ inline
AscendC::MarkStampImpl<pipe, idx>()         [impl:      asc/impl/basic_api/kernel_prof_trace.h]
    │
    ▼ inline
__asc_aicore::asc_mark_stamp<pipe, idx>()   [impl:      asc/impl/utils/debug/asc_aicore_time_impl.h]
    │
    ▼ inline
bisheng::cce::mark_stamp<pipe>(idx)         [bisheng:   tools/bisheng_compiler/.../__clang_cce_aicore_functions.h]
    │
    ▼ 编译器内建展开
[ pipe_barrier(p)   ]  ← dav-c310 对 MTE 系列自动注入（详见 §4）
CCE_SCALAR(__dfx_region)(idx, pipe);   ← Scalar 流水上的硬件 trace 指令
```

整条链全部是 `inline` 模板，不产生函数调用开销；最终落到一条
`__dfx_region(id, pipe)`——本质是 scalar 流水上的一条硬件指令，把
`(scalar_cycle, pipe_tag, id)` 三元组写入 SoC DFX trace 总线。

### 3.2 `__dfx_region(id, pipe)` 的语义

1. `__dfx_region` 是 **scalar 流水上**的指令。scalar 执行到它的那一瞬间，
   往 DFX trace bus 发一条事件 `(cycle, pipe_tag, region_id)`。
2. `pipe_tag`（即模板参数 `p`）**只是分类标签**，方便 viewer 把事件按 pipe
   分组——**不是**延迟到 pipe `p` 实际执行某条指令时才采样。
3. **采集到的 cycle 反映的是 scalar 派发到此处时的时刻**，不是 pipe `p` 当时
   的真实进度。
4. `region_id` 范围 **0–4095**（12 bit，编译期 `static_assert`）。
5. CANN 9.x 头文件中 `mark_stamp_check_pipe<p>()` 会做编译期 `static_assert`，
   拒绝在该核类型上不合法的 pipe：
   - **MIX（AIC+AIV 都定义）**：`S / M / V / MTE1 / MTE2 / MTE3 / FIX`
   - **VEC-only**（`__DAV_VEC__`）：`S / MTE2 / V / MTE3`
   - **CUBE-only**（`__DAV_CUBE__`）：`S / M / MTE1 / MTE2 / MTE3 / FIX`

---

## 4. dav-c310 的自动 `pipe_barrier` 规则

CANN 9.x 的 `bisheng::cce::mark_stamp` 模板对 dav-c310（`__NPU_ARCH__ == 3510`）
做了一项硬件 erratum 规避：**MTE 系列**调用 `mark_stamp` 时自动前置
`pipe_barrier(p)`。具体规则：

| 编译目标 | 自动注入 `pipe_barrier` 的 pipe |
| -------- | ------------------------------- |
| `dav-c310-cube`（`__DAV_CUBE__`） | `MTE1` / `MTE2` / `MTE3` |
| `dav-c310-vec`（`__DAV_VEC__`） | `MTE2` / `MTE3` |
| `dav-c310-mix`（`__DAV_VEC__` 且 `__DAV_CUBE__`） | `MTE1` / `MTE2` / `MTE3` |

**V / M / S / FIX 没有自动 barrier**——要在 V/M 上拿到接近 pipe 真实完成
时刻的 cycle，**必须用户手动**在 `MarkStamp` 之前写
`pipe_barrier(PIPE_V)`（或 `PIPE_M`）。

不加 barrier 也能用——只是事件 cycle 反映的是 "scalar 派发到此 stamp 的时刻"，
而不是 "对应 pipe 跑到此处的时刻"。两种语义都有用：

| 写法 | cycle 含义 |
| ---- | ---------- |
| `MarkStamp<PIPE_V, 7>()`（无前置 barrier） | scalar 派发到此处的时刻；零额外扰动 |
| `pipe_barrier(PIPE_V); MarkStamp<PIPE_V, 7>();` | V pipe 已排空到此处的时刻 |

**barrier 是观测者效应的代价**：原本 V 与 MTE2 可重叠的并行时序会被强行串
行化，画出来的泳道图与未插桩的真实 kernel 有偏差。`mark_stamp` 适合**粗粒度
阶段标记**（前后段间隔远大于 barrier 引入的扰动），不适合**精细测量 pipe 间
重叠率**——加 barrier 抹掉重叠、不加 barrier 取不到 pipe 真实进度，两难。

---

## 5. 数据通路总览：从 stamp 指令到 viewer

```text
┌──────────────────────────────────────────────────────────┐
│  AICore Scalar 流水：                                    │
│    [pipe_barrier(p)]    ← 用户手插或 MTE 自动             │
│    __dfx_region(id, p)  ← scalar 此刻往 DFX trace bus 发  │
│                          一条 (cycle, pipe_tag, id) 事件  │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼ 硬件直接写入
┌──────────────────────────────────────────────────────────┐
│  SoC DFX Trace 模块（独立硬件，与 GM/L2 互不挤占带宽）    │
│      └──► SoC trace ring buffer（driver 管理，落点见 §7.7）│
└──────────────────────────┬───────────────────────────────┘
                           │
                           │ driver 暴露的 prof_* / halProf* 接口（§7）
                           ▼
┌──────────────────────────────────────────────────────────┐
│  任意 consumer 进程：可以是 msprof daemon，也可以是       │
│  任何应用自己用公开头文件接口写的采集线程                 │
└──────────────────────────────────────────────────────────┘
```

两个工程上重要的点：

- **DFX trace bus 是一条相对独立的硬件通路**——与 kernel 自己往 GM 写
  `sys_cnt` 的软件方案彼此独立，这是它相对软件方案的带宽优势所在。但 trace
  数据的**物理落点**（片上 trace SRAM / 保留 device 内存 / trace FIFO）
  **无法从本机头文件与二进制确认**，故本文不再断言"不走 GM"，诚实边界见 §7.7。
- **trace ring buffer 不能直接 mmap**——必须经过 driver 提供的
  `prof_*` / `halProf*` 接口才能拿到。但这些接口是**公开头文件里声明的**，
  任何应用都可以用，并不是 msprof 专属（driver 内部真实读取实现见 §7.6）。

---

## 6. mark_stamp 的 4 字节 chunk 二进制结构

`__dfx_region(id, pipe)` 每执行一次，硬件直接往 trace ring buffer 写
**一个 4 字节 chunk**，结构如下（来自 msprof 解析侧
[`profiling_bean/biu_perf/biu_perf_bean.py`](file:///home/pyptouser/zhengchao/workspace/msprof/analysis/profiling_bean/biu_perf/biu_perf_bean.py)
的 `BiuPerfInstructionBean`）：

```text
bit 31  28 27          16 15                  0
 ┌────────┬───────────────┬──────────────────────┐
 │   pt   │   region_id   │      sys_cnt[low]     │   ← 一条 instruction chunk
 └────────┴───────────────┴──────────────────────┘
     ▲           ▲                   ▲
     │           │                   └── 相对上一个 START_STAMP 的 16-bit delta
     │           └── 用户 mark_stamp 的 12-bit id（events）
     └── ctrl_type / pipe tag（4 bit）
```

> **字段位置以 msprof 解析侧为准**：`ctrl_type` 在高 4 bit（`(word>>28)&0xF`，
> 见 `_get_ctrl_type` 取 `chunk[3]>>4`），`region_id`/events 在 bit 27..16
> （`biu_perf_data[1] & 0xFFF`），`sys_cnt` delta 在低 16 bit
> （`biu_perf_data[0]`）。落地解码器 `l0_decode_chunks`
> （`include/common/l0_perf_profiling.h`）按此实现，并有
> `tests/ut/cpp/a5/test_l0_decode.cpp` 锁定。

`ctrl_type` 枚举（来自
[`msparser/biu_perf/biu_perf_chip6_parser.py`](file:///home/pyptouser/zhengchao/workspace/msprof/analysis/msparser/biu_perf/biu_perf_chip6_parser.py)）：

| ctrl_type | 含义 |
| --------- | ---- |
| 0 | SU（scalar） |
| 1 | VEC（PIPE_V） |
| 2 | CUBE（PIPE_M） |
| 3 | MTE1 |
| 4 | MTE2 |
| 5 | MTE3 |
| 6 | FIXP（PIPE_FIX） |
| 14 | `START_STAMP`——后跟 4 个 chunk 拼成 64-bit 绝对 base cycle + 16-bit block_id |
| 15 | `STATE`——pipe 活动状态位掩码（与 mark_stamp 无关，硬件 PMU 周期采样） |

普通 chunk 里 `sys_cnt` 字段是**相对于上一个 `START_STAMP` 的 delta cycle**
（16 bit，硬件按需周期性插入 `START_STAMP` 重定基准）。consumer 需累加重建
绝对 cycle。

ring buffer 还有两个哨兵：`0xEFFFFFFF` 是块结束标志，`0xDDDDDDDD` 是其后
的对齐填充。

---

## 7. 驱动暴露的接口：两套角色

CANN 把 trace ring 上的事件流抽象成 "profiling channel"，channel 两端是
**producer 和 consumer 两套独立 API**。两套 API **都在公开头文件里有声明**：

- [`include/driver/ascend_hal_base.h`](file:///usr/local/Ascend/cann-9.1.T500/aarch64-linux/include/driver/ascend_hal_base.h)
- [`include/driver/ascend_inpackage_hal.h`](file:///usr/local/Ascend/cann-9.1.T500/aarch64-linux/include/driver/ascend_inpackage_hal.h)

### 7.1 Producer 侧（数据源——硬件 + driver 内部）

谁能产生 profiling 数据、想把它放到 channel 里，就走 producer 侧。

| 接口 | 作用 |
| ---- | ---- |
| `halProfSampleRegister(dev, chan, &para)` | 注册回调（`start_func / sample_func / flush_func / stop_func`）——告诉驱动 "这个 channel 的数据由我提供" |
| `halProfSampleDataReport(...)` | 主动把一段数据推到 channel ring 里 |
| `halProfQueryAvailBufLen(...)` | 查 ring 里还剩多少可写 |

`sample_func` 的签名表明 producer 是被驱动**回调**填数据的，不是直接读数据
的。**软件 producer 才会走这套**（典型用例：CANN runtime 注册"runtime 自己产
生的 launch/dispatch 事件"）。

对硬件 mark_stamp 来说，**producer 是硬件 + driver 内部**——`__dfx_region`
直接写 ring，不需要用户态注册任何回调。因此 mark_stamp 场景下 producer 侧
API **完全不需要调用**。

### 7.2 Consumer 侧（读数据的人——任何应用都可以）

| 接口 | 作用 |
| ---- | ---- |
| `prof_drv_get_channels(dev, &list)` | 枚举该 device 上所有 channel（按名字 + id） |
| `prof_drv_start(dev, chan, &start_para)` | 启动该 channel 的采集 |
| `prof_channel_poll(&info, n, timeout)` | 等任意 channel 有数据可读 |
| `prof_channel_read(dev, chan, out_buf, buf_size)` | 真正把数据从 ring 拉出来 |
| `prof_stop(dev, chan)` | 停止 channel |
| `halProfDataFlush(dev, chan, *len)` | 提示驱动 "把当前积累的数据立刻推上来"——对软件 producer 的延迟优化；硬件 producer 上通常返回 `DRV_ERROR_NOT_SUPPORT` |

`prof_channel_read` 是 consumer 拉数据的主入口。**msprof daemon 走的就是
这条**——见 `libprofimpl.so` 的反汇编符号：`prof_channel_poll` /
`prof_channel_read` / `prof_drv_get_channels` / `prof_drv_start` /
`prof_stop`，全部是 consumer 侧符号。

`halProfDataFlush` 的"`Function not supported`" 字符串在 `libprofimpl.so`
里也确实出现——证实部分 channel 对 flush 是 no-op，consumer 完全可以只
靠 read 拿数据。

### 7.3 consumer 接口只在 host driver 里有真实实现；device stub 是空桩

| 库 | 路径 | 文件大小 | consumer 接口实现 |
| -- | ---- | -------- | ----------------- |
| host driver | `/usr/local/Ascend/driver/lib64/driver/libascend_hal.so` | ~2.9 MB（3048208 B） | **真实实现**（落到 §7.6 的 ring 读取逻辑） |
| device stub | `cann-9.1.T500/aarch64-linux/devlib/device/libascend_hal.so` | 38 KB（38496 B） | **空桩**（符号在、函数体只 `mov w0,#0; ret`） |

⚠️ **此前"device stub 接口完整 → AICPU 可直接读 channel"的推断是错的。**
device-side stub **导出了** `prof_channel_read` / `prof_channel_poll` /
`prof_drv_start` / `prof_stop` 这些符号，但反汇编证实它们全是空桩——函数体只
存下入参、无条件 `mov w0, #0` 返回，**一个字节都不读**（符号仅为满足 device
侧用户程序的链接需求）。真正的 consumer 实现**只存在于 host driver**（§7.6），
device 侧靠这个 stub 拿不到任何数据。AICPU/AICore 能否读到 trace 见 §7.7。

producer 侧 (`halProfSampleRegister` 等) 在 device-side stub 里同样不存在——
但对 mark_stamp 本就不需要，因为硬件本身就是 producer。

### 7.4 channel 命名

driver 内部按字符串命名 channel；biu_perf 类 channel 的名字以
`biu_perf_` 为前缀（在 `libascend_hal.so` 的 strings 里可见）。运行时通
过 `prof_drv_get_channels` 枚举即可拿到 channel_id，**不需要写死**。

### 7.5 实测必需的 host 启动序列（2026-05-26 验证 / 2026-05-27 架构对齐）

只调 consumer 侧接口（`prof_drv_start` + `prof_channel_read`）**不够**。
driver 必须先被告知"当前进程是 instr-profiling consumer"，才会把 DFX
trace ring 的数据 route 到 biu_perf 11..28 这些 consumer channel；否则
`prof_drv_start` 全部 ret=0，`prof_channel_read` 永远返 0 字节。

**所有 L0 专属 driver 握手都集中在
[`L0PerfCollector::initialize`](../src/host/l0_perf_collector.cpp) 内部**
一次完成，与 L2/PMU/dump 同形（DeviceRunner 只调 `collector_.initialize()`，
唯一例外是 `aclrtSetDeviceResLimit` 这条 device-level 调度调用留在 DeviceRunner）：

```cpp
// DeviceRunner::init_l0_perf — device-level only:
aclrtSetDeviceResLimit(dev, ACL_RT_DEV_RES_CUBE_CORE,   kBiuPerfNumGroups);     // 6
aclrtSetDeviceResLimit(dev, ACL_RT_DEV_RES_VECTOR_CORE, kBiuPerfNumGroups * 2); // 12
l0_perf_collector_.initialize(dev, prof_alloc_cb, nullptr, prof_free_cb, output_prefix);
```

`L0PerfCollector::initialize` 内部顺序（任何一步失败回滚已获取资源）：

```cpp
// 1. ACL 层 profiling session（msprof 隐式做的事）
aclprofInit(<output_prefix>/acl_prof_l0, len);
auto *cfg = aclprofCreateConfig(
    {dev}, 1,
    ACL_AICORE_PIPE_UTILIZATION,    // ← 唯一与 biu_perf 语义对得上的 metric
    nullptr,
    ACL_PROF_AICORE_METRICS | ACL_PROF_TASK_TIME);
aclprofStart(cfg);

// 2. ★关键★：底层 instr-profiling 开关。msprof CLI 的 --instr-profiling=on
//    最终就是调这个；缺它 biu_perf 读 0 字节。
MsprofCommandHandle cmd{};
cmd.profSwitch  = 0x00800000ULL;      // PROF_INSTR
cmd.devNums     = 1;
cmd.devIdList[0]= dev;
cmd.modelId     = 0xFFFFFFFFUL;
cmd.type        = 1;                  // PROF_COMMANDHANDLE_TYPE_START
rtProfSetProSwitch(&cmd, sizeof(cmd));

// 3. dlopen libascend_hal + dlsym prof_drv_start/_read/_stop

// 4. 分配 GM marker-queue 头（AICPU push L0TaskFinMarker 用）

// 5. prof_drv_start 18 个 biu_perf channel（chan 11..28）。eager、同步——
//    返回时 HW DFX trace ring 已经在记录，kernel launch 前一定完成。
//    CCU/STARS/AICPU 9 个 prereq channel 已被步骤 1 的 aclprofStart 内部
//    接管，**不要也不能**手动 prof_drv_start（会返 EBUSY=30）。
for (g, sub) prof_drv_start(dev, biu_perf_chan_id(g, sub), &biu_user_data);
```

运行期 drain（与 L2 同形，**不需要额外线程**）：AICPU 把 `L0TaskFinMarker`
push 到 GM → ProfilerBase 的 `mgmt_thread` mirror GM、forward marker →
ProfilerBase 的 `collector_thread` 上 `on_buffer_collected(marker)` 被调用
→ 同一线程上直接 `prof_channel_read` 该 marker 对应 group 的 3 个 sub-core,
decode chunks append 到 records。**driver 不强制 start/read/stop 同线程**
（msprof 自己就是 start 在一个线程 read 在多个不同线程上，shim 实测验证）。

teardown 顺序（`L0PerfCollector::finalize` 内部，对称于 initialize）：

```cpp
// 1. ProfilerBase::stop() — 等 mgmt + collector 线程退出
// 2. prof_stop 18 个 biu_perf channel（每个 ~3-4s，合计 ~60s 是主要开销）
// 3. dlclose libascend_hal
// 4. rtProfSetProSwitch(PROF_INSTR, STOP)
// 5. aclprofStop / aclprofDestroyConfig / aclprofFinalize
// 6. 释放 GM
```

驱动行为细节：marker drain 时 **只有每个 group 的第一个 marker** 拿出全
量数据（典型 50KB–1MB），后续 marker 返 0（ring 已被掏空）——这是因为
driver 用"事件触发 batch flush"模型，不是流式。不需要额外 sweep / poll。

**数据量方差**：同一个 st 多次跑 record 数差异可达 10×（例如 paged_attention_unroll
观察到 75k vs 726k）。原因是：
1. biu_perf 只覆盖 6 个物理核（phys `{0,9,17,18,27,35}`）——scheduler 把
   task 放到其他核上时该 task 不产数据
2. driver 的 batch flush 时机带随机性，不同 run 中"幸运 marker"拿到的
   ring 内容多寡不同
3. 高 START_STAMP 控制 chunk 占比 → records/byte 比下降

这是 HW + driver 行为，不是 host 端能修的。

### 7.6 `prof_channel_read` 在 driver 内部到底从哪读（host 反汇编实证）

把 host 端 `libascend_hal.so` 的 `prof_channel_read` 一路反汇编追到底，read
路径里**没有任何 `ioctl` / `mmap` / `read` 系统调用**——它是一次从进程内
已映射好的环形缓冲区里的 `memcpy`：

```text
prof_channel_read(dev, chan, out_buf, size)          [公共接口]
  → prof_core_chan_read → prof_chan_read
       prof_chan_get_mng(dev,chan) → mng 上下文
       pthread_mutex_lock；检查 mng[+48] started 标志（0→返回 -4）
       ★虚分派★ mng->ops(+64)->read_fn(+40)(dev, chan, &desc, mng->priv+72)
  → 按 channel 注册的后端 ops 三选一：
       prof_hdc_chan_read    （HDC / PCIe 传输，biu_perf 走这条）
       prof_urma_chan_read   （URMA / 统一总线 RDMA 传输）
       prof_user_chan_read   （进程内本地 ring，软件 producer 用）
  → 三者都落到 prof_buff_read → prof_buff_copy（memcpy_s + 原子读写指针）
```

ring 内存布局也是确定的：`prof_buff_get_buf_addr(ring) = ring + 0x1000`，即
头 4KB 是元数据（读/写指针、size），其后才是数据区；`prof_buff_read` 用读写
指针做环形拷贝（绕回分两段）。

**HDC（Host-Device Communication）= driver 提供的 host↔device 通用消息通道**
（句柄 `HDC_SESSION` 等，传输可选 PCIe 或 socket，epoll 事件驱动：
`HDC_EPOLL_DATA_IN`）。对 biu_perf：device 侧产生的 trace 经 HDC 推到 host，
host 侧由一条**事件驱动的接收线程**填 ring：

```text
prof_drv_start → prof_hdc_chan_start
  ├─ prof_hdc_start → prof_hdc_start_msg_send  （把 start 配置发给 device）
  └─ prof_event_start → prof_create_receive_thread(pthread)
        → prof_handle_hdc_events:
             halHdcRecv()            ← 等 device 发来的 HDC 消息
             → prof_hdc_chan_report → prof_buff_write   ← 写进 host ring
  （prof_channel_poll 等的就是接收线程在数据落地时 post 的信号量）
```

**填 ring 的节奏由 device 决定、host 被动收**：host 接收线程是 epoll 收到
`DATA_IN` 才 `halHdcRecv` 一把、填一次 ring。device 何时推，取决于它对片上
trace 缓冲的批量攒够阈值 / 收到 flush / stop 收尾——从 host 视角是**非确定的
批量投递，不是流式**。这正是 §7.5 那两个现象（"第一个 marker 拿全量、后续返
0"、"同 st 多跑 record 数差 10×"）的根因。

host 能调的旋钮，对 mark_stamp 基本不顶用：

| 旋钮 | 在哪设 | 对 biu_perf 是否有效 |
| ---- | ------ | -------------------- |
| `prof_start_para.real_time`（`PROF_REAL`/`PROF_NON_REAL`） | `prof_drv_start` | 仅粗粒度模式开关（边攒边推 / 攒到 stop 批量交付），改不了具体批次时刻 |
| `prof_start_para.sample_period` | `prof_drv_start` | **无效**：库内校验串 `No sample_func` 证明它只对注册了 `sample_func` 的软件 producer 生效；mark_stamp 是硬件事件驱动、无 sample_func |
| `halProfDataFlush` | 运行期 | host 侧虽接到 `prof_hdc_flush_msg_send`，但 §7.5 实测该 channel 返 `DRV_ERROR_NOT_SUPPORT`，**flush 不动** |

一句话：device→host 的填 ring 节奏对 mark_stamp **基本不可控**；你真正能控制
的只是 consumer 端的 drain 时机（何时调 `prof_channel_read`），而多 drain 并不
产生数据，只能拉到 device 已推上来的那一批。

### 7.7 device 侧 trace 落在哪、AICPU/AICore 能不能读（诚实边界）

这是最常被追问的一点，结论要分清"已证实"与"无法证实"：

- **mark_stamp 只能写、不能读**：bisheng / asc 头文件里只有 `__dfx_region`
  （写一条 trace 事件）这一个 intrinsic，**没有任何"读回 trace"的 AICore
  intrinsic**。AICore 程序无法把自己写出去的 stamp 再读回来。
- **device 侧没有暴露 trace 的读 API**：device stub（38KB）里 prof_* 全是空桩
  （§7.3），且**没有任何 biu / instr 的 producer 或 reader 符号**。device 侧
  用户程序（AICPU kernel）链接的就是这个 stub，拿不到真实读实现。
- **物理落点无法从本机材料确认**：头文件与二进制里**没有任何符号或结构**
  指明 biu_perf trace ring 的物理区域——是片上 trace SRAM、保留的 device 内存
  （HBM/DDR）、还是 trace FIFO 寄存器，**目前没有数据能区分**。头文件里的
  `MEM_TYPE_REG_SRAM` / `MEM_TYPE_REG_DDR` 是**异常转储用的 chip DFX 寄存器
  dump**，与 biu_perf 指令 trace 不是一回事，不能拿来当落点证据。

因此对"AICPU / AICore 能否在 device 侧直接读到 trace"的诚实回答是：**用现有
公开 API 与可分析的二进制，读不到、也定位不到**——既无 AICore 读 intrinsic，
也无 device 侧 reader API，更没有暴露给 device 侧的 ring 地址描述符（base/size）。
唯一被验证可用的读路径，是 host 侧经 HDC 的 consumer 链（§7.6）。要进一步坐实
物理落点，需要 dav-c310 的硬件 trace/BIU 手册或 device 侧 driver 源码，这些都
不在本机可见范围内。

---

## 8. msprof 自己用这些接口的方式（仅供对照）

仅用作"业界标准做法"参照，与本项目实施无关。

```text
msprof CLI (tools/profiler/bin/msprof)
    fork application
    └─ load libprofimpl.so
       └─ DrvInstrProfileStart(dev, biu_perf_chan_id, ...)   ← 内部 = prof_drv_start
       └─ loop:
            prof_channel_poll(&info, 1, timeout)
            prof_channel_read(dev, biu_perf_chan_id, buf, sz)
            写 PROF_<ts>/device_<n>/data/biu_perf_<g>_<core>.slice
       └─ DrvStop                                            ← 内部 = prof_stop

msprof analyze
  按 4-byte chunk 切片 + delta cycle 重建 → PROF_*/sqlite/biu_perf.db
  两张表：
   - BiuData          (group, core_type, block_id, ctrl_type, events, base_syscnt)
   - BiuInstrStatus   (..., instruction, timestamp, duration, checkpoint_info)
```

MindStudio / Perfetto viewer 把表里数据按 `(group_id, core_type, ctrl_type)`
分组画核内泳道；相同 `events`（即 region_id）自动同色。

---

## 9. mark_stamp 能做什么 / 不能做什么

### 能做

- ✅ **粗粒度阶段标记**——把一个 task 内部拆成几个有名字的段（例如
  "prologue / matmul / vector / epilogue"），看每段在墙钟上的位置和耗时。
- ✅ **跨 task 同段比较**——同一个段在不同 task 间的耗时差异。
- ✅ **顺序事件的相对排序**——同一 task 内多个 stamp 的先后关系与间隔。
- ✅ **配合 `pipe_barrier(p)` 反映 pipe 完成时刻**——但代价是抹掉该 pipe
  与其它 pipe 的并行重叠。

### 不能做

- ❌ **不加 barrier 时反映 pipe 真实进度**——cycle 全部是 scalar 派发时刻。
- ❌ **观测 pipe 间重叠率**（"V 与 MTE2 重叠了多少 cycle"）——加 barrier
  抹掉重叠、不加 barrier 时间戳又不代表 pipe，两难。要测重叠需要 PMU 或
  专门的硬件计数器，不在 `mark_stamp` 能力范围内。
- ❌ **替代 PMU 性能调优**——pipe stall、cache miss、bank conflict 这些
  通过 mark_stamp 都看不到。

### 工程小贴士

- 区域 id（0–4095）建议按"全局阶段编号"统一编排：例如 `0..15` 留给
  prologue/epilogue，`16..63` 留给主算子的各子阶段，剩下分给调试用。
  相同 id 在 MindStudio 中自动同色，方便跨 task 视觉对齐。
- MTE 系列自带 barrier，可以"随手就插"；V/M 上的 barrier 是显式扰动，
  最好只插在 task 边界 / 阶段边界，不要每条 vec 都包一层。
- `MarkStamp<>` 全部 inline，关掉它的方式是用户自己用宏门控调用点（例如
  `#ifdef SIMPLER_TRACE`）——CANN 这一侧没有提供"运行期开关"，所以
  开/关需要重编 kernel。

---

## 10. 参考

CANN 9.x 关键头文件路径（9.1.T500 验证）：

- 公共接口：`aarch64-linux/asc/include/basic_api/kernel_prof_trace_intf.h`
- 接口 impl：`aarch64-linux/asc/impl/basic_api/kernel_prof_trace_intf_impl.h`
- MarkStampImpl：`aarch64-linux/asc/impl/basic_api/kernel_prof_trace.h`
- `asc_mark_stamp`：`aarch64-linux/asc/impl/utils/debug/asc_aicore_time_impl.h`
- bisheng `mark_stamp` / `__dfx_region`：
  `tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_aicore_functions.h`
  （搜索 `mark_stamp` 与 `__NPU_ARCH__ == 3510`）
- 驱动 prof 接口：
  - `aarch64-linux/include/driver/ascend_hal_base.h`（`prof_drv_get_channels` /
    `prof_drv_start` / `prof_channel_poll` / `prof_channel_read` /
    `prof_stop` / `halProfDataFlush`）
  - `aarch64-linux/include/driver/ascend_inpackage_hal.h`
    （`halProfSampleRegister` / `halProfSampleDataReport` /
    `halProfQueryAvailBufLen` + `prof_sample_*` 结构体）

驱动 / CANN 二进制（仅供分析）：

- host：`/usr/local/Ascend/driver/lib64/driver/libascend_hal.so`
  - §7.6 read 链可复现的符号：`prof_channel_read` → `prof_core_chan_read`
    → `prof_chan_read` →（虚分派）`prof_hdc_chan_read` / `prof_urma_chan_read`
    / `prof_user_chan_read` → `prof_buff_read` → `prof_buff_copy`；HDC 接收侧
    `prof_handle_hdc_events` → `halHdcRecv` → `prof_hdc_chan_report` →
    `prof_buff_write`（`nm` / `objdump -d` 即可验证）
- device-side stub：`cann-9.1.T500/aarch64-linux/devlib/device/libascend_hal.so`
  （`prof_channel_read` 等为空桩，§7.3）
- CANN 包装层：`aarch64-linux/lib64/libprofimpl.so`（`DrvInstrProfileStart` 等）
- msprof daemon：`tools/profiler/bin/msprof`

外部参考（msprof 项目源码）：

- chunk 二进制结构：
  [`profiling_bean/biu_perf/biu_perf_bean.py`](file:///home/pyptouser/zhengchao/workspace/msprof/analysis/profiling_bean/biu_perf/biu_perf_bean.py)
- `.slice` 解析完整流程：
  [`msparser/biu_perf/biu_perf_chip6_parser.py`](file:///home/pyptouser/zhengchao/workspace/msprof/analysis/msparser/biu_perf/biu_perf_chip6_parser.py)

CANN 版本：

- 当前环境安装路径：`/usr/local/Ascend/cann-9.1.T500/`
- 版本号即目录名 `9.1.T500`（T 系列预发布；顶层 `version.info` 留空，
  详细版本见 `aarch64-linux/ascend_toolkit_install.info`）
- 原撰写基线：CANN 9.0.0（`version=26.0.rc1`），原机器路径在本机上不存在
