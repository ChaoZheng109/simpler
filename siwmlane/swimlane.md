# json文件数据结构定义
merged_swimlane.json 是一个 Chrome Trace Event Format (Perfetto 格式) 的 JSON 文件，主要结构如下：

{
  "traceEvents": [
    // 进程元数据
    {
      "args": {"name": "Machine View"},
      "cat": "__metadata",
      "name": "process_name",
      "ph": "M",
      "pid": 1
    },
    
    // 线程元数据（每个 AI Core）
    {
      "args": {"name": "AIC_0"},
      "cat": "__metadata",
      "name": "thread_name",
      "ph": "M",
      "pid": 1,
      "tid": 1002
    },
    
    // 任务执行事件（Duration Event）
    {
      "args": {
        "event-hint": "Task:[seq-func-op], rootHash:xxx, ...",
        "ioperand-hint": "输入操作数信息",
        "ooperand-hint": "输出操作数信息",
        "execution-hint": "执行时间分析",
        "color": "颜色标签",
        "taskId": 原始任务ID,
        "seqNo": 序列号
      },
      "cat": "event",
      "id": 事件ID,
      "name": "任务名称",
      "ph": "X",  // Duration event
      "pid": 1,
      "tid": 线程ID,
      "ts": 开始时间戳,
      "dur": 持续时间
    },
    
    // 任务依赖关系（Flow Event）
    {
      "cat": "machine-view-last-dep",
      "id": 事件ID,
      "name": "machine-view-last-dep",
      "ph": "s",  // Flow start
      "pid": 1,
      "tid": 源线程ID,
      "ts": 源时间戳
    },
    {
      "bp": "e",
      "cat": "machine-view-last-dep",
      "id": 事件ID,
      "name": "machine-view-last-dep",
      "ph": "f",  // Flow finish
      "pid": 1,
      "tid": 目标线程ID,
      "ts": 目标时间戳
    },
    
    // 就绪队列计数（Counter Event）
    {
      "name": "ReadyCount_AIC",
      "pid": 1,
      "tid": 1,
      "ph": "C",  // Counter
      "ts": 时间戳,
      "args": {"size": 就绪任务数}
    },
    
    // 内存使用情况（如果是动态拓扑）
    {
      "name": "Ideal_Mem_Usage(Task)",
      "pid": 1,
      "tid": 1,
      "ph": "C",
      "ts": 时间戳,
      "args": {"/byte": 内存字节数}
    },
    {
      "name": "OOO_Mem_Usage(UB1)",
      "pid": 1,
      "tid": 1,
      "ph": "C",
      "ts": 时间戳,
      "args": {"/byte": 内存字节数}
    }
  ]
}


# 主要数据内容
文件包含以下几类数据：
1. 进程/线程元数据 - 定义 AI Core 的显示结构
2. 任务执行事件 - 每个任务的开始时间、结束时间、执行核心
3. 任务依赖关系 - 任务之间的数据流依赖（flow events）
4. 就绪队列计数 - AIC/AIV 的就绪任务数量变化
5. 依赖求解速率 - Dependence Solving (MHz)
6. 内存使用情况 (仅动态拓扑):
    Ideal_Mem_Usage(Task) - 理想内存使用
    OOO_Mem_Usage - 乱序执行的内存使用（按内存类型分类）