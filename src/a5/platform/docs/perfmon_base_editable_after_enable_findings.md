# AICore perfmon:固件使能后 base_addr 仍可改写实验小结

**平台**:a5 / dav-c310 / CANN 9.1.T500 ·**用例**:paged_attention_unroll(108 AICore)·**device 3**·**两次复现一致**

**目标(#905 衍生)**:验证一种更轻的接入方式 —— AICPU 不再 blind-config 全部
perfmon 寄存器,而是**等固件 kickstart 使能后,只改 `PERF_MON_BASE_ADDR`**,把
回写指向自管 GM,其余(`en` / `global_en` / `buf_len`)全用固件默认。回答一个
前置疑问:`base_addr` 在 `perf_mon_en` 已经为 1 之后还能不能写进去。

## 结论

**能。固件使能(`en=0x5`)之后,`base_addr` 仍可被 AICPU 改写,且硬件 DMA 会
改用新 base 落数。**

- 寄存器层面:**108/108 核**写后回读 == 写入值,无一例外,包括 `en=0x5 gen=0x1`
  正在写的核。spec 里"`base` 仅在 `en=0` 时可写"在 dav-c310 实测**不成立**。
- 数据层面:base 被改的 18 个激活核,`wptr` 从改写时的 ~0xe3020 继续涨到 ~1.6M,
  finalize 时 base 仍是我们的 → 数据进了自管 buffer;同时 driver channel
  `prof_channel_read` 读回 **0 records**(数据被从 channel 抢到了自管 buffer)。

## 复现方式

```bash
PYPTO_L0_PERFMON_ADDR_ONLY=1 L0_SKIP_PROF_SET_SWITCH=1 L0_SKIP_ACLPROF_START=1 \
  python tests/st/a5/tensormap_and_ringbuffer/paged_attention_unroll/test_paged_attention_unroll.py \
  --device <d> --platform a5 --log-level v0 --enable-l0-swimlane --build
```

- `PYPTO_L0_PERFMON_ADDR_ONLY=1`:跳过 launch 前 blind-config + host 不等
  `perfmon_ready`,改为在 scheduler 握手(`handshake_all_cores`,kickstart 已跑)
  后,每核**只**写 `BASE_ADDR_L/H`。代码:`perfmon_aicpu_set_addr_after_handshake`。
- 必须保留 `--enable-l0-swimlane`:其 ③ `prof_drv_start` 触发 kickstart 设 `en`。
- **AICPU 日志在 `log/debug/device-<d>/`,不在 host 重定向的 `host_log.txt`。**

## 关键证据(同一握手函数内,写 base 前后各读一次)

```
core 0: DEFAULTS base=0x3ff_b4a7f000  | wrote base=0x1000_00400000 -> readback base=0x1000_00400000
core 9: DEFAULTS base=0x3ff_a8a7f000  | wrote base=0x1000_00d00000 -> readback base=0x1000_00d00000
...(18 个 gen=1 核,写前均为 0x3ff_xxx,写后均为我们的 0x1000_xxx)
```

finalize(整跑结束)再读:18 核 base 仍 `0x1000_xxx`,`wptr` 已涨到 ~1.6M。

两跑均 **108/108 核写后回读 == 写入值,0 失败**。

## 二次复现:残留已排除

第二跑(64MB buffer)结果与首跑一致,并额外坐实三点:

1. **`0x3ff_xxx` 是固件每跑新分配,不是静态残留** —— 同一核两跑读到的固件 base
   不同:`core 0` `0x3ff_b4a7f000` → `0x3ff_c8000000`,`core 9` `0x3ff_a8a7f000`
   → `0x3ff_a5b80000`。残留会逐位相同;变了 ⇒ driver 每跑重切 ring。18/18 监控核
   两跑都被刷成 `0x3ff_xxx`;90 非监控核仍是上一跑 addr-only 写的 `0x1000_xxx` 残留
   (driver 不碰它们,所以它们才是真残留)。
2. **64MB 治好溢出**:18 核 finalize 最大 `wptr=2326688`(~2.2MB)≪ 64MB,无 wrap。
3. **channel 仍 `total_records=0`**,数据始终在自管 buffer。

## 三类读 base 的时间点

| 读取点 | 何时 | 18 监控核 | 含义 |
| --- | --- | --- | --- |
| DEFAULTS(写前) | 握手,kickstart 后 | `0x3ff_xxx` | 固件/driver 留的 base |
| readback(写后) | 握手,写后即读 | `0x1000_xxx` | 写成功、立即生效 |
| finalize | 整跑结束 | `0x1000_xxx` | 整跑一直是我们的,数据已落 |

## "写前 base = 0x3ff_xxx 是固件的"——推断依据(未读 driver 源码)

1. 该值是 kickstart 之后、addr-only 写之前读到的(本轮 AICPU 还没写)。
2. 不是我们的残留(已二次坐实):若是残留,同一核两跑应读到逐位相同的值;实测
   `0x3ff_xxx` 逐跑变化(见"二次复现"),且 18 监控核每跑都被刷成 `0x3ff_xxx`,
   而 90 非监控核稳定停在我们上一跑写的 `0x1000_xxx`(那才是真残留,driver 不碰)。
3. 地址区间:`0x3ff_xxx` 是设备高地址保留区,18 个值按 64MB 等距递减(`0xb4a7f000,
   0xb0a7f000, 0xaca7f000...`)= driver 给 18 条 biu_perf channel 连续切的 ring;
   用户态 `rtMalloc` 只给 `0x1000_xxx`。

## 边界与注意

- **激活条件是 `en` 且 `gen` 同时为 1**(`PERF_MON_EN` AND `PERF_MON_GLOBAL_EN`)。
  这一跑里 kickstart 把 `en=0x5` 设到了全 108 核,所以**差异落在 `gen`** 上:只有
  driver 注册了 channel 的 18 核 `gen=0x1`,其余 90 核 `gen=0x0` → 静止。
  addr-only 不写 gen,故若要全 108 核出数据,需在 addr-only 里补写 `global_en=1`
  (同 blind-config)。
  - 这 18 核 = host 侧报的 **6 个 AICore 簇 × 每簇 3 core(1 AIC-cube + 2 AIV-vector)**
    = 18 物理 core。host 的"monitors 6 AICores / retire 30 / block_dim 6"是**簇粒度**,
    device perfmon 寄存器是**物理 core 粒度**,`6×3=18`、`(6+30)×3=108`,两边一致,
    非 bug。
- **`buf_len` 仍是固件默认 `0x3fffffc`(≈64MB)**:addr-only 不改它,故自管 buffer
  必须 ≥64MB,否则 HW 按 64MB 写会越界踩进相邻核。已把
  `device_runner.cpp::kPerfmonBufBytes` 调到 `1u<<26`(64MiB);108×64MB≈6.9GB HBM。
- 没 reset `WPTR_O`:数据从 `base + 旧wptr`(~0xe3020)起写,buffer 开头会留旧值/
  哨兵;要从头干净写需另写 `WPTR_O=0`。

## 待解

- `base_addr` 在 `en=1` 时可写已确认;`buf_len` / `wptr` 在 `en=1` 时是否同样可写
  尚未单独验证(下一步:addr-only 里加写 `BUF_LEN` + `WPTR_O=0` 观察是否 stuck)。
- `0x3ff_xxx` 是否确为 driver channel ring 基址,可在 `start_all_biu_perf_channels`
  打印 `prof_drv_start` 返回的 ring 地址对照坐实。
