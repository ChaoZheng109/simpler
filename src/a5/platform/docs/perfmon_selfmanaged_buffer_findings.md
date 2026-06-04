# AICore perfmon 自管 buffer:三接口实验小结

**平台**:a5 / dav-c310 / CANN 9.1.T500 ·**用例**:paged_attention_unroll(108 AICore)

**目标(#905)**:AICPU 直接编程 perfmon 寄存器(`0xB000` 段),把回写
`base_addr` 指向自管 GM,跑完 `rtMemcpy` 取回,绕开 driver biu_perf channel。

## 结论

**只调 ③ `prof_drv_start` 就够**:perfmon 被使能(`perf_mon_en` bit0=1),
数据按我们预置的 `base_addr` 写进自管 buffer,**全 108 核**(远超 biu_perf
固定的 6 核)。① `rtProfSetProSwitch`、② `aclprofStart` 对使能和落点**均无关**。

## 三个接口

| 接口 | 定义 / 层 | 实际动作 |
| --- | --- | --- |
| ① `rtProfSetProSwitch` | `api_c.cc:1457` / runtime | 置 `BiuperfProfFlag` → SQE `featureFlag\|=BIUPERF`;并下一条空事件配置的 PROFILING_ENABLE 给 TS |
| ② `aclprofStart` | `prof_acl_api.cpp:244` / ACL·msprof | 经 msprof 把 AICORE 事件译成 profSwitch,最终也汇入 ①,下带事件配置的 PROFILING_ENABLE |
| ③ `prof_drv_start` | `prof_interface.c:90` / driver | 注册 biu_perf channel 消费者(host 侧 4MB ring)。**此动作触发 HWTS 在 kickstart set `perf_mon_en`** |

## 实验结果

三个接口任意组合,实测 `perf_mon_en` 与自管 buffer 落数据情况:

| ② | ① | ③ | `perf_mon_en` | 自管 buffer |
| :-: | :-: | :-: | --- | --- |
| ✗ | ✓ | ✗ | 0x4 | 无 |
| ✓ | ✓ | ✗ | 0x4 | 无 |
| ✗ | ✗ | ✓ | **0x5** | **有(108 核)** |
| ✗ | ✓ | ✓ | 0x5 | 有 |
| ✓ | ✗ | ✓ | 0x5 | 有 |
| ✓ | ✓ | ✓ | 0x5 | 有 |

```
en=0x5 ⟺ ③ 在场      （①② 有无均不影响）
```

## 为什么只有 ③ 能打通

`perf_mon_en` 不是 AICPU 软件写一下就生效的——AICPU 盲配里的 `en=1` 会被
HWTS 的 **kickstart**(AICore 调度起来时由固件配寄存器)覆盖。kickstart 是否
把 `en` set 成 1,**判据是"该 device 有没有注册 biu_perf channel 消费者"**,
而这正是 ③ `prof_drv_start` 干的事。

- ①②**只影响 SQE 标记 / TS 事件配置,不参与 kickstart 的 en 判据** → 有无都不翻 `en`。
- ③ 一注册,HWTS kickstart 就 set `en`,且**保留**我们预置的 `base_addr`
  (只动 en、不动 base),于是 perfmon HW 照着我们的 base 写。

因此 ③ 是使能的充要条件,落点天然是我们的自管 buffer。

## 落地方案(#905 实现蓝图,不改其他仓接口)

启动序列:host `prof_drv_start`(触发使能)→ AICPU 盲配 regs(base/buf_len)→
handshake 同步 → host launch AICore。channel 数据不消费,结束 `prof_stop`。

**per-task 增量 drain(不清零),挂在 `scheduler_completion.cpp` 的 task FIN
处**(`l0_perf_aicpu_complete_record` 旁,AICPU 已持有 task id / core / cycle
window)。每 task、每核:

1. 读该核 `samp_wrt`(0xB028)+ `wptr`(0xB01C);
2. `delta = samp_wrt - last_samp_wrt[core]`;
3. `delta ≥ buf_len` → **溢出**:log 报错 + 上报 host(drop/截断,仿 dump
   tensor),`delta` 截到 `buf_len`;
4. 按 `wptr` 从 perfmon buffer 拷 `delta` 字节(跨界拆两段)到我们的 profiling
   buffer,打 task id + core 标签;
5. `last_samp_wrt[core] = samp_wrt`;
6. **全程只读寄存器**——不清零、不写 `wptr`、不碰 `en`(写这些需 en=0,代价高
   且与固件 kickstart 状态冲突)。

host 收"带 task id 的原始记录"→ 解析 → json(原始 vs 解析放 host,AICPU 不解析)。

**开放项(后补,先不阻塞性能)**:DMA 完成同步——task FIN 时该核 perfmon 样本
可能有在途 DMA 未落 buffer。暂不加 barrier,先打通,后续按实测决定是否需要轻量
flush;尾部漏算的样本会并入下一个 task,误差可接受。

**约束**:per-task 产出 < `buf_len`(确定成立);超出按溢出处理。

## 待解(给 HWTS 固件 / driver)

落点已通,剩一个解耦问题:`en` 目前绑死 ③,而 ③ 带 4MB host ring +
每通道 ~3-4s(共 ~60s)teardown 的包袱(这些 channel 数据我们其实不消费)。

- **HWTS**:能否让 kickstart 用更轻的信号(如我们预置的 `perf_mon_global_en`)
  直接 set `en`,免去 channel 注册?并确认 kickstart 始终保留我们的 `base_addr`。
- **Driver**:有无 `prof_drv_start` 的轻量变体,只触发设备侧使能、不分配 host
  ring、不引入 teardown?并确认 6 核限制只是 host channel 概念(实测设备侧全
  108 核都写)。
