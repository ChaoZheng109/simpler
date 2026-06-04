# A5 perfmon 初始化同步方案

**平台**:a5 onboard / dav-c310
**关联**:[perfmon_register_check.md](perfmon_register_check.md)(寄存器编程与现象)

## 1. 为什么需要同步

perfmon 的 `perf_mon_en` 是 kickstart 寄存器,需要在 **AICore kernel 被发射
(kickstart)之前**完成全部配置(base_addr / buf_len / en 等),发射时配置才
就绪。

当前实现中,perfmon 寄存器由 AICPU 在 **scheduler dispatch 阶段**(handshake
之后)编程,而 AICore kernel 与 AICPU 在不同 stream 上并发 launch,**无法保证
perfmon 配置先于 AICore kickstart 完成**。

约束(已确认):

- perfmon 寄存器是 device 侧 MMIO,`halResMap(target = PROCESS_CP1)` 映射给
  AICPU 进程,**host 无法直接写**;只能由 AICPU 写。
- AICore kernel 的 launch(`rtKernelLaunchWithHandleV2` / `<<<>>>`)是 host
  API,**AICPU 无法 launch AICore**。

因此配置(AICPU)与发射(host)分属两端,必须在两者之间建立同步点。

## 2. 死锁与破解

朴素做法会死锁:

```
perfmon 配置  需要  physical_core_id(逻辑核↔物理核映射)
physical_core_id 需要  handshake
handshake     需要  AICore 已 launch
AICore launch 需要  perfmon 配置先完成   ← 环
```

**破解:盲配所有物理核,绕开 `physical_core_id`。**

- `physical_core_id` 仅用于"逻辑 worker ↔ 物理核"的映射,handshake 后才有。
- 但 host 的 `regs[]` 是 `halResMap` 按物理核 `core_idx = 0..N` 逐个枚举建的,
  **launch 之前即已知全部物理核 reg_base**,不依赖 handshake。
- perfmon 是 per-physical-core 硬件。对**所有物理核**直接配置,即无需逻辑↔物理
  映射,也无需 handshake。

代价:未参与本轮 task 的物理核也开了 perfmon,其 buffer 空置(无害)。

## 3. 同步时序

```
host:   init_perfmon_probe          分配 per-core buffer + 填 KernelArgs
host:   launch AICPU
AICPU:  盲配 regs[0..N] 全部物理核    base/buf_len/wptr/glitch/en,不依赖 handshake
AICPU:  写 GM ready flag = 1
host:   poll ready flag(D2H 自旋,带超时)
host:   launch AICore kernel         此刻 perfmon 已就绪,kickstart 时配置完整
AICore: handshake → AICPU 派发 task   照常
```

`regs[]` 与 perfmon buffer table 均在 launch 之前经 KernelArgs 备好,AICPU 一
启动即可盲配。

## 4. 实现要点

1. **盲配版 init**:`perfmon_aicpu_init` 由"按逻辑核(`physical_core_ids`)"改为
   "遍历 `regs[]` 全部物理核";host 侧 buffer table 按物理核 index 对应。
2. **时机前移**:perfmon 配置从 scheduler dispatch 阶段移到 AICPU 启动早期
   (handshake 之前),单线程执行一次(如 `simpler_aicpu_init`)。
3. **ready flag**:host 在 `init_perfmon_probe` 额外分配一个 GM flag;AICPU 配完
   写 1;host 在 launch AICPU 之后、launch AICore 之前 poll 该 flag。
4. **finalize 不变**:仍由 AICPU 在 shutdown 阶段读回计数 + 关闭 en。

## 5. 前提风险

本方案保证 **perfmon 配置在 AICore kickstart 之前就绪**(解决时序),但不解决
"软件写 `perf_mon_en` 是否等价于硬件 kickstart 置位"这一问题(见
[perfmon_register_check.md](perfmon_register_check.md) §5.6)。若 en 必须由硬件
发射动作置位、软件预写无效,则本方案仍不足以让 HW 产生 trace。
