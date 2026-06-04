# 给 CANN runtime 团队:如何通过 kernel launch 启用 AICore perfmon

**平台**:a5 / dav-c310 / CANN 9.1.T500
**一句话**:我们想在 launch AICore kernel 时让 HWTS 在 kickstart 阶段启用该核
的 perfmon(`perf_mon_en`),但当前 launch 接口找不到传 biuperf 标志的入口。

## 1. 我们在做什么

绕开 driver 的 biu_perf channel(`prof_drv_start` / `prof_channel_read`)通路,
改由我们自己:

- AICPU 直接编程 AICore 的 perfmon 寄存器(`0xB000` 段),把回写 buffer 的
  `base_addr` 指向我们自管的 GM,跑完自己 `rtMemcpy` 取回;
- 目的是去掉 channel 通路的 6 核限制、批量投递不可控、~60s `prof_stop` 等问题。

## 2. 现状:launch 方式 + perfmon 配置

**AICore kernel launch**(常驻 while-loop kernel,aic + aiv 编进同一个 ELF):

```c
rtRegisterAllKernel(&elf_binary, &handle);   // ELF 注册,拿 handle
rtTaskCfgInfo_t cfg = {}; cfg.schemMode = RT_SCHEM_MODE_BATCH;
rtKernelLaunchWithHandleV2(handle, /*tilingKey=*/0, block_dim,
                           &args, nullptr, stream, &cfg);
```

**perfmon 寄存器**:AICPU 在 kernel launch 之前对每个物理核写好
`base_addr_l/h`、`buf_len`、`glitch_filter`、`global_en`、`perf_mon_en=1`。

## 3. 现象:唯独 `perf_mon_en` 被 kickstart 刷掉

实测(AICore 在 kernel entry 读自己寄存器回传)证实:

| 寄存器 | launch 前(我们写的) | AICore entry(kickstart 后) |
| --- | --- | --- |
| `base_addr_l/h` | per-core(自管 buffer) | **保持我们的值** |
| `buf_len` / `glitch` / `global_en` | 我们的值 | **保持我们的值** |
| **`perf_mon_en`** | `0x1` | **`0x4`**(bit0 被清) |

即:**kickstart 只把 `perf_mon_en` 改掉了(没碰 base_addr)**。结合调度同事的
说明 —— HWTS 在 swap-in task 时从 swapbuf 读 `biu_perf_en` 决定该核是否记录
biu_perf —— 我们理解为:**launch 没带 biuperf 标志 → swapbuf 的 `biu_perf_en`
= 0 → kickstart 时把 `perf_mon_en` 清掉**。

我们也试过 `rtProfSetProSwitch(PROF_INSTR, START)`:调用返回 0,但
`perf_mon_en` 仍是 `0x4`,无数据 —— 它不经过这条 per-core perfmon 的使能。

## 4. 待 runtime 团队确认的问题

我们在 `runtime/kernel.h` 找到:

```c
// STARS topic scheduler sqe : topic_type
#define RT_KERNEL_DEVICE_FIRST (0x10U)
#define RT_KERNEL_HOST_ONLY    (0x20U)
#define RT_KERNEL_HOST_FIRST   (0x40U)
#define RT_KERNEL_BIUPERF_FLAG (0x80U)   // ← 这个
```

但 driver 源码里看不到 `RT_KERNEL_BIUPERF_FLAG` → SQE → swapbuf `biu_perf_en`
的映射(应在 libruntime 闭源实现内)。请确认:

1. **`RT_KERNEL_BIUPERF_FLAG` 是不是用来"让 HWTS 在 kickstart 时启用该核
   perfmon(set `perf_mon_en`)"的标志?** 置上它之后,HWTS 是否会保留我们预先
   写进寄存器的 `base_addr`(只 set en、不改落点)?

2. **这个 flag 通过哪个 launch 接口、哪个参数传?**
   - `rtKernelLaunchWithHandleV2` 的入参(`tilingKey` / `rtArgsEx_t` /
     `rtTaskCfgInfo_t`)里我们没找到 flag 入口;
   - 带 `flags` 参数的只有 `rtKernelLaunchWithFlag` / `WithFlagV2`,但其 `flags`
     文档注释写的是 "dump flag",且它们用 `stubFunc` 而非 handle。
   - **`flags` 参数是否接受 `RT_KERNEL_BIUPERF_FLAG`?** 还是另有途径?

3. **我们的 kernel 是 `rtRegisterAllKernel`(ELF)+ handle、aic/aiv 同一 ELF 的
   常驻 kernel。如果必须走 `WithFlagV2`(stubFunc),这种 ELF/handle 注册的
   kernel 怎么拿到 `stubFunc`(`rtFunctionRegister` / `rtGetFunctionByName`)?
   aic + aiv 两个符号在 stubFunc 路径下怎么分发?** 还是说 handle 路径有办法
   带这个 flag(例如某个我们漏看的接口/字段)?

## 5. 已排除项(供参考,避免重复建议)

- base_addr 类型:确认是 device VA,我们写的 VA 在 kernel entry 仍保留 ✓
- 内存池:HBM / DDR / DDR_NC 三种,`rtMalloc` 返回同一 VA,与现象无关 ✓
- 寄存器编程:`base/buf_len/wptr/glitch/global_en` 全部 readback 正确 ✓
- 时序:已保证 perfmon 配置在 AICore launch 之前完成(AICPU 配好→host 轮询
  ready flag→再 launch AICore)✓
- `rtProfSetProSwitch(PROF_INSTR)`:不使能这条 per-core perfmon ✓

核心未决就一个:**怎么从 launch 把 biuperf 标志传进 SQE,让 HWTS kickstart 时
set `perf_mon_en`(而非清掉),且保留我们的 base_addr。**
