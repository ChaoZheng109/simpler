# A5 perfmon 寄存器编程现状

**主题**:perfmon 寄存器配置就绪状态下,HW 未产生 DMA 写。记录编程序列、寄存器
语义、实验现象与已收敛结论
**平台**:dav-c310 / CANN 9.1.T500 / `__NPU_ARCH__ = 3510`

## 1. 摘要

**目标**:由 AICPU 直接编程 AICore 的 perfmon 硬件寄存器,把 trace 回写到我们
自管的 GM buffer,从而绕开 driver 的 biu_perf / msprof 通道。

**现象**:perfmon 配置寄存器全部编程并 readback 一致、两个 enable 全程为 1,
但 HW 的"产生计数"`perf_mon_samp_crt` 与"写出计数"`perf_mon_samp_wrt` 始终为
0,回写 buffer 除诊断哨兵外全为 0 —— **HW 未产生任何 trace、未写一字节**。

**文档结构**:第 2–4 节是配置事实(基址/缓冲区、寄存器清单、编程序列);
第 5 节是实验与现象(各实验给出取证与结论);第 6 节汇总已排除的故障模式;
第 7 节汇总仍未决的核对点。

## 2. 平台基址与缓冲区

| 项 | 取值 |
| --- | --- |
| AICore MMIO 映射 | `halResMap(RES_AICORE)` 提供每核 3 MB 连续 MMIO;其 `map_addr` 即 perfmon 窗口基址(已确认,5.3) |
| 每核 `reg_base` | 由 host 在 init 阶段构建的 `regs_array[physical_core_id]`,AICPU 经 `get_platform_regs()` 取回 |
| 回写 buffer 分配 | host 端 `rtMalloc(size = 0x100000, mem_type, 0)`,每核独立;`mem_type` 已分别尝试 `RT_MEMORY_HBM` / `RT_MEMORY_DDR` / `RT_MEMORY_DDR_NC`,均返回同一地址,详见 5.4 |
| `buf_addr` 类型 | rtMalloc 返回的 device **VA**(`perf_mon_base_addr` 期望 VA,已确认,5.4) |
| Core 0 实测 `buf_addr` | `0x100000200000` |
| 每核 `buf_len` | `0x00100000`(1 MiB) |

## 3. 寄存器清单

所有 offset 均相对所属 AICore 的 `reg_base`,32 位 MMIO 访问。

### 3.1 编程写入

| 名称 | offset | 位域 | 写入值 | 语义(依据现有 spec) |
| --- | ---: | --- | --- | --- |
| `perf_mon_en` | `0x00C4` | bit 0 | `0x1` | perfmon 实例使能;spec 称其为 kickstart 寄存器,正常由硬件调度器在 task kickstart 时置位、task 结束自动清除(详见 5.6) |
| `perf_mon_global_en` | `0xB000` | bit 0 | `0x1` | per-physical-core 使能;与 `perf_mon_en` **同时为 1** HW 才 trace |
| `perf_mon_buf_len` | `0xB004` | [31:0] | `0x00100000` | buffer 长度(byte);仅 en=0 时可写 |
| `perf_mon_glitch_filter` | `0xB008` | [3:0] | `0x0` | busy/idle 宽度小于此值的信号其 trace 被 cancel;0 = 不过滤 |
| `perf_mon_base_addr_l` | `0xB00C` | [31:0] | `buf_addr & 0xFFFFFFFF` | 48-bit **VA** 回写基地址低 32 位;仅 en=0 时可写 |
| `perf_mon_base_addr_h` | `0xB010` | [15:0] | `(buf_addr >> 32) & 0xFFFF` | 48-bit VA 回写基地址高 16 位;仅 en=0 时可写 |
| `perf_mon_wptr_o` | `0xB01C` | [31:0] | `0x0` | RW 写指针(byte);setup 时写 0 复位,仅 en=0 时可写 |
| `perf_mon_samp_crt_clr` | `0xB024` | bit 0 | `0x1` | 产生计数器清除(写 1 清) |
| `perf_mon_samp_wrt` | `0xB028` | [31:0] | `0x0` | 已写入计数器(RW,写 0 清);仅 en=0 时可写 |

### 3.2 仅诊断读取

| 名称 | offset | 位域 | 用途 |
| --- | ---: | --- | --- |
| `perf_mon_samp_crt` | `0xB020` | [31:0] | finalize 读取,HW 产生数据量(samples)的判据 |
| (上表各写入字段) | — | — | 同时在 init 末尾(仅 core 0)与 finalize 读回,核对是否被覆盖/清除 |

更正记录(相对早期理解):`perf_mon_base_addr` 已确认为 **VA**;`perf_mon_wptr_o`
为 **RW**(早期误判为只读),setup 时须写 0;新增 `perf_mon_glitch_filter`
(`0xB008`)。`perf_mon_en` 是 kickstart 寄存器(早期误判为普通 enable),
影响见 5.6。

## 4. 编程序列

以下为 init / finalize 的完整寄存器写入序列(裸 MMIO,与项目内抽象解耦)。
写入顺序、写入值、offset 均以代码为准。

使能顺序:`perf_mon_global_en`(per-core gate)先置 1,`perf_mon_en`(最后一个
enable 写)后置 1,使 en 在 0→1 时 `global_en` 已为 1、AND 条件当场满足。配置类
寄存器(base/buf_len/wptr/counters)均在 en=0 时写入。

注意:此处对 `perf_mon_en` 的写是 **AICPU 的普通 MMIO 值写入**,并非 spec 所述
"hardware scheduler 在 task kickstart 时置位"的那个硬件动作。软件值写入与硬件
kickstart 置位是否等价,是本文未决问题之一(见 5.6)。

```c
/* 寄存器 offset 命名(与第 3 节一一对应) */
#define PERF_MON_EN            0x00C4u  /* bit0  perfmon 实例使能(spec: kickstart 寄存器) */
#define PERF_MON_GLOBAL_EN     0xB000u  /* bit0  per-core 全局使能 */
#define PERF_MON_BUF_LEN       0xB004u  /* [31:0] buffer 长度(byte) */
#define PERF_MON_GLITCH_FILTER 0xB008u  /* [3:0]  glitch filter,0 = 不过滤 */
#define PERF_MON_BASE_ADDR_L   0xB00Cu  /* [31:0] VA 回写基地址低 32 位 */
#define PERF_MON_BASE_ADDR_H   0xB010u  /* [15:0] VA 回写基地址高 16 位 */
#define PERF_MON_WPTR_O        0xB01Cu  /* [31:0] RW 写指针;setup 写 0 */
#define PERF_MON_SAMP_CRT      0xB020u  /* [31:0] 产生数据计数 */
#define PERF_MON_SAMP_CRT_CLR  0xB024u  /* bit0  产生计数清除(写 1 清) */
#define PERF_MON_SAMP_WRT      0xB028u  /* [31:0] 已写入计数(写 0 清) */

#define WR32(base, off, val) (*(volatile uint32_t *)((base) + (off)) = (uint32_t)(val))
#define RD32(base, off)      (*(volatile uint32_t *)((base) + (off)))

void perfmon_init_one_core(uint64_t reg_base, uint64_t buf_addr, uint32_t buf_len)
{
    /* (0) 诊断哨兵:事后判定 HW 是否覆盖 buffer */
    *(volatile uint32_t *)buf_addr = 0xDEADBEEFu;

    /* (1) 关闭两级 gate(清除潜在残留状态) */
    WR32(reg_base, PERF_MON_GLOBAL_EN, 0);
    WR32(reg_base, PERF_MON_EN,        0);

    /* (2) 编程 48-bit 回写基地址(仅 en=0 时) */
    WR32(reg_base, PERF_MON_BASE_ADDR_L, (uint32_t)(buf_addr & 0xFFFFFFFFu));
    WR32(reg_base, PERF_MON_BASE_ADDR_H, (uint32_t)((buf_addr >> 32) & 0xFFFFu));

    /* (3) 编程 buffer 长度(仅 en=0 时) */
    WR32(reg_base, PERF_MON_BUF_LEN, buf_len);

    /* (4) 复位写指针 + 关闭 glitch filter(仅 en=0 时) */
    WR32(reg_base, PERF_MON_WPTR_O,        0u);
    WR32(reg_base, PERF_MON_GLITCH_FILTER, 0u);

    /* (5) 清零两个 byte 计数器(仅 en=0 时) */
    WR32(reg_base, PERF_MON_SAMP_CRT_CLR, 1u);  /* 写 1 清产生计数 */
    WR32(reg_base, PERF_MON_SAMP_WRT,     0u);  /* 写 0 清已写入计数 */

    /* (6) 打开 per-core gate */
    WR32(reg_base, PERF_MON_GLOBAL_EN, 1u);

    /* (7) 写屏障,保证以上写入对 HW 可见后再置 en */
    asm volatile("dsb sy" ::: "memory");

    /* (8) 最后置 en —— 普通 MMIO 值写入,非硬件 kickstart(见正文) */
    WR32(reg_base, PERF_MON_EN, 1u);
}

void perfmon_finalize_one_core(uint64_t reg_base)
{
    /* 诊断读回 */
    uint32_t wptr   = RD32(reg_base, PERF_MON_WPTR_O);
    uint32_t s_wrt  = RD32(reg_base, PERF_MON_SAMP_WRT);
    uint32_t s_crt  = RD32(reg_base, PERF_MON_SAMP_CRT);
    uint32_t base_l = RD32(reg_base, PERF_MON_BASE_ADDR_L);
    uint32_t base_h = RD32(reg_base, PERF_MON_BASE_ADDR_H);
    uint32_t blen   = RD32(reg_base, PERF_MON_BUF_LEN);
    uint32_t en     = RD32(reg_base, PERF_MON_EN);
    uint32_t gen    = RD32(reg_base, PERF_MON_GLOBAL_EN);
    /* ... 落日志 ... */

    /* 与 init 反序:先撤 trigger,再关 gate */
    WR32(reg_base, PERF_MON_EN,        0);
    WR32(reg_base, PERF_MON_GLOBAL_EN, 0);
}
```

## 5. 实验与现象

### 5.0 现象一句话

全部 perfmon 配置寄存器写入后,init 直后与 kernel 结束后的 readback 均等于
写入值、两个 enable 全程为 1,但 `perf_mon_wptr_o` / `perf_mon_samp_wrt` /
`perf_mon_samp_crt` 始终为 0,回写 buffer 除 AICPU 哨兵外全为 0 —— **HW 未
产生任何 DMA 写**。

运行环境:2026-06-02;device 3;`block_dim` 对应 108 个 AICore 全部初始化;
kernel = `vector_mul_stamp`(含若干 `AscendC::MarkStamp<PIPE_V/PIPE_M, id>()`)。

### 5.1 寄存器值实测(core 0,init 直后与 finalize 各读一次)

| 寄存器 | offset | 写入 | init readback | finalize readback |
| --- | ---: | --- | --- | --- |
| `perf_mon_base_addr_l` | 0xB00C | `0x00200000` | `0x00200000` | `0x00200000` |
| `perf_mon_base_addr_h` | 0xB010 | `0x00001000` | `0x00001000` | `0x00001000` |
| `perf_mon_buf_len` | 0xB004 | `0x00100000` | `0x00100000` | `0x00100000` |
| `perf_mon_glitch_filter` | 0xB008 | `0x0` | `0x0` | — |
| `perf_mon_en` | 0x00C4 | `0x1` | `0x1` | `0x1` |
| `perf_mon_global_en` | 0xB000 | `0x1` | `0x1` | `0x1` |
| `perf_mon_wptr_o` | 0xB01C | `0x0` | `0x0` | `0` |
| `perf_mon_samp_wrt` | 0xB028 | (清 0) | — | `0` |
| `perf_mon_samp_crt` | 0xB020 | (清 0) | — | `0` |

core 0 实测 `buf_addr` = `0x100000200000`。108 个 AICore 结果一致。

### 5.2 回写 buffer 内容

每核 1 MiB buffer 经 host D2H 拷回后:byte[0..3] = `EF BE AD DE`
(init 时 AICPU 写入的 `0xDEADBEEF` 哨兵,little-endian),byte[4..]
全为 `0x00`。哨兵保留 → AICPU 写 GM 路径正常;其余全 0 → perfmon HW 未写。

---

以下各节是本次工作的核对实验(A–D),分别检查现状中几个最可疑的输入。
各节末尾给出该项的结论或未决问题;5.6 为当前主要未决项。

### 5.3 实验 A —— 寄存器基地址 `reg_base` 的来源与正确性

**`reg_base` 来源**:host 侧 `halResMap` 逐核映射,`res_type = RES_AICORE`,
每核映射长度 `REG_AICORE_MAP_SIZE = 0x300000`(3 MB);返回的 `map_addr` 即该
**物理 AICore 的 MMIO 基址**,写入每核 `regs_array[]`。AICPU 侧
`get_platform_regs()[physical_core_id]` 取回同一值作为 perfmon 写寄存器的
`reg_base`。源码:`src/a5/platform/onboard/host/host_regs.cpp:31-86`。

**关键细节(AIC 与 AIV 的 base 不同)**:一个物理 AICore = 1 个 AIC + 2 个 AIV
子核,三者共享同一次 `halResMap` 的 `map_addr`,但:

| 子核 | reg_base |
| --- | --- |
| AIC | `map_addr` |
| AIV #0 | `map_addr + REG_AIV_FIRST_OFFSET`(`+1 MB`) |
| AIV #1 | `map_addr + REG_AIV_SECOND_OFFSET`(`+2 MB`) |

(`REG_SUB_CORE_STRIDE = 1 MB`,见 `host_regs.h:43-49`。)perfmon offset
`0xB000` 当前对**三类子核一视同仁**地加在各自 `reg_base` 上。

**结论(已确认)**:`halResMap(RES_AICORE)` 的 `map_addr` 即 perfmon 寄存器
窗口基址,无需额外窗口选择 / 偏移。本项排除。

### 5.4 实验 B —— 回写 buffer 地址 `buf_addr` 的来源与异常

**`buf_addr` 来源**:host 端 `rtMalloc(0x100000, mem_type, 0)`,每核一块,
device 地址填入 `perf_mon_base_addr_l/h`。

**异常现象**:为排查"perfmon DMA 仅接 DDR、HBM 段 VA 不可达"的假设,用三种
`mem_type` 各跑一次(其余完全一致,每次 run 前已 `rtFree`):

| `rtMalloc` mem_type | addr table VA | core 0 `buf_addr` | finalize `wptr/samp_wrt/samp_crt` | buffer |
| --- | --- | --- | --- | --- |
| `RT_MEMORY_HBM` (0x2) | `0x100000175000` | `0x100000200000` | `0 / 0 / 0` | 仅哨兵 |
| `RT_MEMORY_DDR` (0x4) | `0x100000175000` | `0x100000200000` | `0 / 0 / 0` | 仅哨兵 |
| `RT_MEMORY_DDR_NC` (0x20) | `0x100000175000` | `0x100000200000` | `0 / 0 / 0` | 仅哨兵 |

**三种 mem_type 在独立 run 间返回逐字节相同的 VA**。推断:dav-c310 用户态
`rtMalloc` 实际仅暴露**单一内存池**,`RT_MEMORY_DDR` / `_DDR_NC` 在 runtime
内部退化为 `RT_MEMORY_HBM`。已尝试的替代分配途径同样不可行:`rtMallocPhysical`
(`runtime/mem.h:1149`)的 `rtDrvMemHandle` 不暴露 PA,仅能经 `rtMemMap` 绑回 VA。

**补充复验**:上述三种 mem_type 在早期(寄存器配置不全时)各跑过一次;在 5.6
的配置补全(wptr=0 / glitch=0 / 使能顺序修正)之后,`RT_MEMORY_DDR` 与
`RT_MEMORY_HBM` 又各复跑一次,结果不变(同一 VA、`0/0/0`、仅哨兵)。**内存池
变量已彻底排除**——与 HW 不写无关。

**结论(已确认)**:`perf_mon_base_addr` 期望填入的是 **device VA**(非物理
PA),`rtMalloc` 返回的 VA 即可。地址来源与类型本项排除——HW 不写与 `buf_addr`
无关。

(因此前述"是否需 PA / SMMU streamid 不一致 / 不应由用户给地址"等疑问均不成立;
内存池单一化现象仅是 a5 用户态行为,对本问题无影响。)

### 5.5 实验 C —— 寄存器 offset 与使能流程的正确性

**offset 来源**:第 3 节寄存器的 offset / 位域来自现有 spec 口头同步,
**尚未与最新寄存器手册逐一核对**。

**已补全的配置**(基于后续 spec 澄清,见第 4 节最新序列):

- `perf_mon_wptr_o`(`0xB01C`)setup 时写 0 复位(早期遗漏);
- `perf_mon_glitch_filter`(`0xB008`)写 0 关闭过滤(早期完全遗漏);
- 使能顺序改为 **先 `global_en` 后 `perf_mon_en`**(早期为先 en 后 global)。

三项补全后重跑,`wptr` / `glitch` readback 均为 0(确认写入生效),但 HW 仍
未写(`samp_crt = 0`)。即:**这三项不是根因,但已是正确配置应保留**。

**未决问题**:

1. 寄存器 **offset 与位域**逐一核对(尤其 `perf_mon_en = 0x00C4`,与 `0xB000`
   段不连续)。
2. `perf_mon_samp_crt_clr`(0xB024 写 1 清)、`perf_mon_samp_wrt`(0xB028 写 0
   清)的清除语义。
3. 使能流程**是否仍有遗漏的必需步骤**(wptr/glitch/顺序已补,仍无数据),例如:
   - 事件选择 / event mask / sample mode / sample period 寄存器(未编程则无
     数据源);
   - 启动 / 触发(kick)寄存器、或 reset deassert / release 序列。
   - AICore 侧 gate,见 5.6。

### 5.6 实验 D —— `perf_mon_en` 的启动机制(当前主要悬而未决项)

**spec 描述**:`perf_mon_en` 是 kickstart 寄存器,**正常由硬件调度器在 task
kickstart 时置位、task 结束自动清除**。

**a5 dispatch 路径调研结论**:a5 的 AICore 任务派发是**纯软件 MMIO 握手,无硬件
调度器**——AICPU 向 `DATA_MAIN_BASE`(SPR `0xD0`)写 31-bit task_id,AICore 跑
**一个常驻 kernel** 在 `aicore_executor.cpp` 死循环 poll 该寄存器执行任务;派发
描述符 `PTO2DispatchPayload` 无任何 `feature_flag` 字段,kernel launch
(`rtKernelLaunchWithHandleV2`)也无 per-task feature 配置。即:**a5 上不存在
"硬件调度器在 kickstart 时帮我们 set `perf_mon_en`"这条路径**,也没有 driver
`ts_stars_aic_aiv_sqe_t.feature_flag` 那种 SQE BIUPERF bit 可设。

**已确认**:`perf_mon_en` **不一定要走 hardware scheduler 的
fast-path kickstart** 来置位——即**软件写 en 是被允许的**,并非必须由硬件调度器
在 task kickstart 时设置。这与 a5 无硬件调度器的事实一致,也排除了"a5 上 perfmon
根本无法软件启动"的担忧。

**由此收敛**:`perf_mon_en` 可由软件置位;已软件写 en=1 且 readback=1,但
HW 仍不 trace。因此问题**不在 en 的触发路径**,而在以下两者之一:

1. **由谁写 en** —— AICPU 从外部写"别核"的 MMIO,是否等价于本核置位?现有 PMU
   的总开关 `GLB_PMU_EN` 是由 **AICore 本核**用 SPR 指令 `set_ctrl()` 打开的
   (`pmu_aicore_begin/end`),而非 AICPU 外部写。`perf_mon_en` 是否也需**本核**
   写才生效?(注:a5 AICore 侧 `write_reg` 仅支持 SPR 类 CTRL/COND,对任意
   MMIO 无写路径,故若需本核写 MMIO,还需确认 AICore 的 device-store 指令。)
2. **数据源未配** —— 该 perfmon block 是否需先经事件 / 采样配置才会产生 sample
   (见 5.7)。

**未决问题**:

1. AICPU 从外部 MMIO 写 `perf_mon_en`,与 AICore 本核置位,二者是否等价?若不
   等价,本核应以何种指令 / 对哪个 SPR 或 MMIO 写?
2. perfmon 启动前是否还需事件 / 采样 / mode 配置(见 5.7)。

### 5.7 背景:数据源接入

`0xB000` 这套 **per-AICore perfmon** 与 SoC 级 **BIU perf**(msprof biu_perf
channel、接 `__dfx_region` / `mark_stamp` 的那块)是否同一 HW 块?其数据源是
`mark_stamp` 指令 trace,还是 per-AICore PMU 周期采样 / 其它?此问决定本方案
能否抓到 `mark_stamp`,以及 5.6 第 3 点的配置内容。

## 6. 已排除的故障模式

| 故障假设 | 排除依据 |
| --- | --- |
| MMIO 写未生效 | init 后立即 readback 等于写入值(5.1) |
| Firmware 抢占 / 覆盖配置寄存器 | finalize readback 与 init 一致(5.1) |
| Firmware 自动关闭 enable | finalize 时两个 enable 仍为 1(5.1) |
| `buf_addr` 对 AICPU 不可写 | AICPU 哨兵成功落地(5.2) |
| `buf_addr` 地址类型(VA/PA) | spec 确认 VA + 哨兵落地(5.4) |
| `buf_len` 未编程 | 已编程 `0x00100000`,readback 一致(5.1) |
| `wptr` 残留非零 | 已写 0,readback=0(5.5) |
| glitch filter 全过滤 | 已写 0,readback=0(5.5) |
| 使能顺序(en/global)颠倒 | 两种顺序均试(5.5) |
| 使能前 barrier(dsb)扰动时序 | 去掉 gate→en 间的 dsb 复跑,行为一致(5.5) |
| HBM 段不可达、需换 DDR 池 | 三种 mem_type、配置不全/补全均试,行为一致(5.4) |
| 缺硬件 SQE feature_flag kickstart | a5 为纯软件 MMIO 派发;en 不必走 fast-path kickstart(5.6) |
| `reg_base` 获取方式错误 | `halResMap` map_addr 即 perfmon 窗口基址,已确认(5.3) |
| `buf_addr` 地址类型(VA/PA)错误 | base_addr 期望 VA,获取方式正确,已确认(5.4) |

## 7. 核对清单(汇总)

软件侧可调变量已逐一排除(第 7 节)。`reg_base` 获取、`buf_addr`(VA)、`en` 软件
置位许可均已确认。当前悬而未决收敛为**两点**:

1. **【最高】`perf_mon_en` 由谁写** —— 软件写 en=1 readback=1 但 HW 不 trace。
   AICPU 从外部写"别核"MMIO 与 **AICore 本核置位**是否等价?(现有 PMU 总开关
   `GLB_PMU_EN` 须由本核 `set_ctrl()` 打开。)若需本核写,以何指令 / 写何处? → 5.6
2. **数据源是否需配置** —— `0xB000` perfmon 接的是 `mark_stamp` 指令 trace 还是
   PMU 周期采样;启动前是否还需事件 / 采样 / mode 配置才产生 sample。 → 5.6 / 5.7

(已确认 OK,不再追问:`reg_base` 获取方式 → 5.3;`buf_addr` 为 VA 且获取方式
正确 → 5.4;`en` 允许软件置位、不必 fast-path kickstart → 5.6。)
