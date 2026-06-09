# AICore perfmon 自管 buffer vs driver channel:实验小结与未解问题

**平台**:a5 / dav-c310 / CANN 9.1.T500 ·**用例**:paged_attention_unroll(108核)
+ vector_mul_stamp(小用例,单 kernel_mul 任务,便于归核)·**device 3**

**目标**:不依赖 driver biu_perf channel,让全 108 核的 perfmon 数据写进我们自管
的 GM buffer(每核 64MB),且 channel 读不到数据。

## 一、已确认的硬件机制(前提)

| 项 | 结论 | 出处 |
| --- | --- | --- |
| 使能依赖 | 必须 host 调 `prof_drv_start`(channel start ③)才能使能 perfmon | 实验一系列 |
| `en` 谁控 | **kickstart 全局设 en**:即使 AICPU 不写 en,kickstart 也把**全 108 核** `en=0x5`(实测) | exp8 |
| `gen` 谁控 | **选择性闸门**:driver 只给它监控的 **18 核**设 `gen=1`;其余靠我们写 | exp4/exp8 |
| 18 核构成 | 6 个 AICore 簇 × 每簇 3 core(1 AIC-cube + 2 AIV-vector)= 18 物理 core | — |
| 激活条件 | `perf_mon_en AND perf_mon_global_en` 同时为 1 | — |
| `base` 可写性 | en=1 时**仍可改写**(108/108 回读一致) | exp2 |
| `buf_len` 可写性 | en=1 时可写;`wptr` en=1 时**写不进**(需 en=0) | exp3 |
| 采样模型 | perfmon **按核连续采**整跑所有 pipe 活动,**不是按任务**。空转核采得最多 | exp8 |
| 数据通路 | **单通路**:每核样本只去 base 当时指向的地方;channel 与我们 buffer 是 disjoint 的核集合,**不重复** | exp8 |

## 二、两种接入方式与现象

在 host 调 `prof_drv_start` 的基础上:

**方式 A —— 握手后只改 addr(exp4 / addr-only)**
- AICPU 与 AICore 同时启动,AICPU 不预配,握手后仅把 base 改成自管 buffer。
- 现象:**只有 driver 使能了 gen 的 18 核有数据**,其余 90 核 `gen=0` 不采。
- 好处:**channel 为空**,这 18 核的数据全从我们的 addr 读到。
- 缺点:覆盖不了全 108 核。

**方式 B —— 握手前盲配(exp6/7/8 / arm-all、gen-only)**
- AICPU 先启动,launch 前盲配:写 `buf_len`、写 `gen=1`(给全 108 核),再启动 AICore。
- 现象:**全 108 核都被使能**(gen 我们写 + en kickstart 全开),都有数据。
- 缺点:**channel 上又出现数据**。

## 三、核心结论

**当前这套临时方案,无法同时满足三点:① 全 108 核出数据 ② 数据全走我们的 addr
③ channel 为空。** 方式 A 满足 ②③ 但只有 18 核;方式 B 满足 ①② 但 channel 漏数据。

根因在于 `en` / `gen` 的控制权:`en` 被 kickstart 全局强开(关不掉),`gen` 是唯一
可选择性控制的闸门 —— 但要让全核出数据就得全核开 gen,而这正是 channel 漏数据的
触发条件(见下)。

## 四、channel 何时有数据 —— 横向对比与规律

| 跑 | 握手前 arm gen | retire | base 写法 | **channel total_records** |
| --- | :-: | :-: | --- | :-: |
| addr-only (paged) | ✗ | on | 握手后 | **0** |
| addr-only (vector 小) | ✗ | on | 握手后 | **0** |
| addr-only + skip-retire (vector 小) | ✗ | off | 握手后 | **0** |
| arm-all exp6 (paged) | ✓ | off | 握手前+后 | 772568 |
| arm-all exp7 (paged) | ✓ | off | 握手后 | 519488 |
| gen-only exp8 (vector 小) | ✓ | off | 握手后 | 744 |

**规律(唯一决定因素)**:

> **channel 有数据 ⟺ 我们在握手前(AICore launch 前)盲配写了 `gen=1`。**
> `base` 写法、`retire` on/off、workload 全部不影响 channel。

- 不预写 gen(方式 A / addr-only,gen 全交给 driver)→ **channel 永远空**。
- 预写 gen(方式 B / arm-all、gen-only)→ **channel 就有数据**。

**机制(timing)——决定权在"gen 变 1 的时刻 vs base 被 redirect 的时刻"的先后**:

- **预写 gen**:监控核在 **kickstart 那一刻就 `en AND gen` 成立、开始采样**,而此刻 base
  还是 driver 的 `0x3ff`(我们只在握手后才 redirect)。`[kickstart → 握手后 redirect]`
  这段窗口的样本落进 driver ring → channel 读到。跑得早、很快结束的任务(kernel_mul)
  整份落在窗口内 → 全漏给 channel(exp8:`phys 36/37` 744 条进 channel,我们 buffer 只
  2336 字节);空转整跑的核样本主要在 redirect 之后 → 进我们 buffer(`phys 0` = 3.3M)。
- **不预写 gen**:在我们 redirect base 之前,监控核没在 `0x3ff` 上产生样本(等真正
  armed/干活时 base 已是我们的)→ 数据进我们 buffer → channel 空。

**一句话**:channel 有没有数据,取决于"监控核在 base 还指向 driver ring(`0x3ff`)的
时候,有没有被 arm 起来采样"。预写 gen 会让它们在 redirect 之前就采 → 喂 channel;
不预写就不会。

**根本矛盾**:用来使能 perfmon 的 `prof_drv_start`,同时把那 18 个监控核在 kickstart 时
绑定到了 driver ring。要让全核出数据必须握手前 arm gen,而这恰恰是喂 channel 的条件。
"全 108 核出数据"与"channel 为空"在当前手段下不可兼得(见三)。

## 五、附带发现:冗余数据问题

方式 B 全核 arm 后,我们 buffer 出现大量冗余:小用例只有 1 个 kernel_mul 任务,buffer
却有 **16.5MB / 108 核**。原因是 perfmon 按核连续采,**空转核(无 kernel 的 AIC cube、
等任务的核)整跑都在 runtime spin/handshake/dispatch loop 里打转,被连续采样**,反而比
真干活的核数据更多(`phys 0` 空转 = 3.3M)。真任务信号(kernel_mul)很小,被噪声淹没。
若要"按任务"的干净数据,需 task 的 cycle 窗口 + core id 把样本归属回任务,而非 arm 全核裸采。

## 六、复现开关(env,全部可选;均需配合 `--enable-l0-swimlane`)

实现:host `device_runner.cpp` 读 env → `PROFILING_FLAG_PERFMON_*` → AICPU
`perfmon_collector_aicpu.cpp`。固定前缀命令:

```bash
<ENV> L0_SKIP_PROF_SET_SWITCH=1 L0_SKIP_ACLPROF_START=1 \
  python <test.py> --device <d> --platform a5 --log-level v0 --enable-l0-swimlane --build
```

| env | 含义 | 握手前盲配 | 握手后写 base | skip retire | channel |
| --- | --- | :-: | :-: | :-: | :-: |
| `PYPTO_L0_PERFMON_PROBE` | 盲配全寄存器(含 base) | 全套 | ✗ | ✗ | 有 |
| `PYPTO_L0_PERFMON_ADDR_ONLY` | 仅握手后写 base(方式 A) | ✗ | ✓ | ✗ | **空** |
| `PYPTO_L0_PERFMON_UNIFY` | addr-only + 握手后补 buf_len/wptr/gen | ✗ | ✓ | ✓ | 有 |
| `PYPTO_L0_PERFMON_ARM_ALL` | 盲配(不含 base)+ 握手后写 base(方式 B) | 除 base | ✓ | ✓ | 有 |
| `PYPTO_L0_PERFMON_GEN_ONLY` | ARM_ALL 但盲配只开 gen、不开 en | 除 base/en | ✓ | ✓ | 有 |
| `PYPTO_L0_PERFMON_SKIP_RETIRE` | 独立,跳过 retire(可与任意组合) | — | — | ✓ | 不影响 |

每核 buffer `kPerfmonBufBytes = 1u<<26`(64MiB);108×64MB≈6.9GB HBM + 同等 .bin。
AICPU 端日志在 `log/debug/device-<d>/`,不在 host `host_log.txt`。

> 注:`PROBE` 走 `simpler_aicpu_init` 全套盲配 + host 等 `perfmon_ready`;`ADDR_ONLY`
> 跳过盲配与等待、握手后写;`ARM_ALL`/`GEN_ONLY`/`UNIFY` 各自的盲配范围见上表
> "握手前盲配"列。
