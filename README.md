## 项目说明：OpenCL MNIST 双路前向推理 + Web 前端

本项目基于 OpenCL 实现 MNIST 手写数字识别，提供两种不一样的前向推理方式，并通过内置 HTTP 服务配合网页画板实现“手写→缩放→前向→展示”的端到端体验。

### 快速概览（我们做了什么、怎么做的）
- 两种前向一套权重一次初始化，支持多次推理并分别计时对比；
- A：单核前向（一个 kernel 完成 3 层 MLP）；
- B：分层前向（统一层 kernel 顺序执行 3 次，前两层含 ReLU）；
- C：Web 服务 + 画板前端，调用 A/B 两种前向并展示预测与耗时；
- 通过 CMake + vcpkg 组织项目与依赖。

### A/B/C 我具体做了什么
- A（`src/A.cpp` + `src/kernels/A.cl`）：把 784→128→64→10 的三层计算合并到一个内核 `mlp_forward_one`，减少内核切换与中间写回；权重启动时一次加载并常驻，推理时只写入输入、执行、读回输出并计时。
- B（`src/B.cpp` + `src/kernels/B.cl`）：实现通用全连接层 `fc_relu_layer`，按层顺序执行三次（前两层启用 ReLU，最后一层关闭）；修复 barrier 早退导致的结果不一致问题（详见 `BUGFIX.md`）；同样一次加载权重、每次推理仅写输入并计时。
- C（`src/web_server.cpp` + `src/web/index.html`）：自研 Winsock HTTP 服务器，GET `/` 返回画板页面；POST `/infer` 接收 784 维像素，分别调用 A/B 两种前向，返回预测与耗时；前端完成手写图像缩放到 28x28、灰度与归一化并展示结果。

### 目标与特性
- 一次初始化、一次加载权重，多次前向推理复用（两种方法均支持）
- 每次输入图像，分别调用两种方法前向并统计耗时
- 内置 Windows HTTP 服务（Winsock），提供 POST `/infer` 推理接口；GET `/` 返回前端页面
- 前端网页内置画板，鼠标/触屏手写数字，自动缩放为 28x28 并归一化为 784 维数组
- CMake 跨平台构建（主要在 Windows + OpenCL 环境验证）

> **📝 项目更新说明**：项目文件结构已重新整理，主要可执行目标重命名为 `A`、`B`、`C`，内核文件移至 `src/kernels/` 目录，前端文件移至 `src/web/` 目录。详细变更请参考 `BUGFIX.md`。

---

## 目录结构与文件职责

### 根目录
- `CMakeLists.txt`：CMake 构建配置，定义三个可执行目标：`A`、`B`、`C`
- `CMakePresets.json`：提供 VS2022 + vcpkg 的 x64-Release 预设
- `vcpkg.json`：依赖声明（`opencl`）
- `README.md`：本文档
- `BUGFIX.md`：Bug修复说明文档

### 源码目录 `src/`
- `opencl_utils.hpp`
  - 常用 OpenCL 工具：错误检查 `checkCLError`、平台/设备选择 `pick_first_device_print`、构建日志 `build_log`

- `A.cpp`
  - OpenCL 前向（单样本）演示程序；读取 `src/kernels/A.cl` 内核，创建缓冲区并完成一次前向

- `B.cpp`
  - 批量/分块并行版本；读取 `src/kernels/B.cl` 的 `fc_relu_layer`，三层依次调用，支持 `--batch N`

- `kernels/A.cl`
  - 单核函数前向：`mlp_forward_one` 在一个 work-item 中完成 784→128→64→10 三层计算

- `kernels/B.cl`
  - 分层核函数：`fc_relu_layer` 支持矩阵-向量乘法 + 可选 ReLU，用于多层串联（批量版本也复用）

- `infer_engine.hpp`
  - 核心抽象与复用：
    - `ModelWeights`：权重、偏置载体
    - `OpenCLContext`：设备/上下文/队列一次初始化
    - `ForwardSingleKernel`：方法A，单核函数前向（`mlp_forward_one`）
    - `ForwardLayerKernels`：方法B，分层核函数三次调用（`fc_relu_layer`）
    - 两种方法均：一次初始化常量权重缓冲区，多次 `infer()` 时仅写入输入并读取输出

- `web_server.cpp`（C目标）
  - Windows 内置 HTTP 服务（Winsock）：
    - GET `/` 或 `/index.html`：返回 `src/web/index.html`
    - POST `/infer`：接收 JSON：`{"pixels":[784个0~1浮点]}`
      - 同时调用方法A、方法B，统计各自耗时，返回预测与耗时
  - 关键流程：
    1) 启动时加载权重（`src/model_out/`），一次初始化 OpenCL 与两种前向引擎
    2) 每个请求仅写入输入 buffer，调用 `infer()` 并读回 logits
    3) 以 `std::chrono::high_resolution_clock` 统计方法A、方法B推理时长

### 前端目录 `src/web/`
- `index.html`
  - 画板：280x280 手写，`drawImage` 缩放至 28x28，读取像素转灰度并归一化到 [0,1]
  - 点击“前向推理”后，POST 到 `/infer`，在页面分别显示两种方法预测类别与耗时

### 模型与数据
- `src/model_out/`
  - `linear0_W.txt`、`linear0_b.txt`、`linear1_W.txt`、`linear1_b.txt`、`linear2_W.txt`、`linear2_b.txt`
  - 文本格式：第一行是 `行 列`（向量为 `len 1`），随后为按行主序的浮点数
- `src/data/mnist/`（样例MNIST原始数据）

---

## 两种前向方法与关键函数

### 方法A：单核函数（`src/kernels/A.cl` → `mlp_forward_one`）
- 特点：把三层计算（含两层 ReLU）放到单一 kernel 内以减小内核切换与中间读写
- 关键缓冲：输入 `x [784]`、权重/偏置常量 buffer、输出 `logits [10]`
- 关键代码：`ForwardSingleKernel::initialize()`、`ForwardSingleKernel::infer()`

### 方法B：分层核函数（`src/kernels/B.cl` → `fc_relu_layer`）
- 特点：统一层计算核函数，三次调用完成三层（前两层 ReLU=1，最后一层 ReLU=0）
- 关键缓冲：`X/H0/H1/Y` 四个中间/输出 buffer；权重/偏置常量 buffer
- 关键代码：`ForwardLayerKernels::initialize()`、`ForwardLayerKernel::infer()`

### 复用与计时
- 权重、偏置在启动时一次性读入并创建 OpenCL 只读缓冲（两种方法均复用）
- 前向调用仅写入 784 输入、设置 kernel 参数并执行，随后读取 10 维 logits
- 计时使用 `std::chrono::high_resolution_clock` 记录 A 与 B 推理的独立耗时

---

## 构建与运行

### 依赖
- Windows + Visual Studio 2022 + CMake
- OpenCL 运行时（GPU 驱动或 CPU OpenCL 驱动）
- vcpkg（可选但推荐），设置环境变量 `VCPKG_ROOT`

### 使用 CMakePresets（推荐）
1) 配置：
```
cmake --preset x64-Release
```
2) 构建：
```
cmake --build --preset x64-Release
```
3) 产物位于：`build/x64-Release/Release/`

### 运行
- 命令行演示：
  - `A.exe [可选: sample.txt]`
  - `B.exe [--batch N] [可选: sample.txt]`

- Web 服务：
  1) 运行 `C.exe`
  2) 浏览器打开 `http://127.0.0.1:8080`
  3) 在画板上手写数字，点击“前向推理”，查看两种方法的预测与耗时

### Web 接口
- POST `/infer`
  - 请求 JSON：
```json
{
  "pixels": [0.0, 0.0, ..., 0.85] // 长度 784，范围 [0,1]
}
```
  - 响应 JSON：
```json
{
  "methodA": { "pred": 7, "ms": 0.42 },
  "methodB": { "pred": 7, "ms": 0.58 }
}
```

---

## 关键实现思路
1) 统一加载权重与 OpenCL 初始化：通过 `OpenCLContext` + 两类前向类的 `initialize()` 实现“一次初始化，多次推理”
2) 两种前向的比较：
   - 方法A 减少 kernel 切换与中间写回，适合轻量小网络
   - 方法B 复用统一层核函数，结构更清晰，易于扩展（如替换激活函数、切换层尺寸）
3) 前端画板像素预处理：缩放至 28x28，并按 `1 - 灰度/255` 转为黑底白字风格，归一化 [0,1]
4) 时序与性能：每次请求独立计时 A、B，避免互相干扰；兼顾可读性与鲁棒性（路径探测、异常报告）

---

## 常见问题（FAQ）
- 无法找到 OpenCL 平台/设备：请安装 GPU 驱动或 CPU OpenCL 运行时
- 构建报找不到 CMake：把 CMake 可执行加入 PATH，或使用 VS 的 CMake 插件/GUI
- 页面空白或 404：确保 `C.exe` 运行后访问 `http://127.0.0.1:8080`；确认构建后 `src/web/index.html` 已被复制到目标目录
- 权重文件找不到：放在 `src/model_out/` 下，或保持与可执行路径的相对位置（项目已做多路径探测）

---

## 版权与许可
本项目仅用于学习与演示目的。模型与数据版权归各自原始作者所有。


