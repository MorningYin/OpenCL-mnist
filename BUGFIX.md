## Bug 说明：分层前向内核与单核前向输出不一致

### 现象
- 对同一输入和同一组权重，方法A（单核 `mlp_forward_one`）与方法B（分层内核 `fc_relu_layer` 三次调用）输出不一致。

### 根因
- 分层内核 `src/kernels/mlp_layer.cl` 为提升性能，采用了 `__local` 缓存与 `barrier(CLK_LOCAL_MEM_FENCE)` 同步。
- 为对齐工作组，`gws[0]` 向上取整到局部维度（如 16 的倍数），会产生 `o >= out_dim` 的“无效”工作项。
- 原实现中，对无效工作项使用 `return` 提前退出。但同一工作组的其他线程会进入 `barrier`，从而造成 barrier 不一致，OpenCL 语义未定义，可能出现随机错误或与方法A不一致的结果。

### 受影响代码
- 文件：`src/kernels/mlp_layer.cl`
- 问题片段：
```
const int o = get_global_id(0);
const int b = get_global_id(1);
if (o >= out_dim || b >= N) return; // 无效线程提前退出，后续还有 barrier
...
barrier(CLK_LOCAL_MEM_FENCE);
...
barrier(CLK_LOCAL_MEM_FENCE);
```

### 修复思路
- 禁止在包含 barrier 的路径中提前 `return`。
- 所有工作项都参与 barrier，同步一致；仅对“有效线程”进行累加、加偏置与写回。
- 具体做法：引入 `valid_o` / `valid_b` 变量，保护累加与写回；协作加载时对无效 `b` 填 0。

### 风险与验证
- 修复后性能影响极小（仅增加条件判断），同步语义正确。
- 验证方式：相同输入下比较方法A与方法B输出，应完全一致；对多个样本验证。

### 变更文件
- `src/kernels/mlp_layer.cl`：移除 early return，添加 `valid_o/valid_b` 守护逻辑。


