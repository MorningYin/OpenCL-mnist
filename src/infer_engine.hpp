// =============================================================
// 高层推理引擎抽象（两种前向方法）
// - 方法A：单核函数前向（src/mlp_forward.cl -> mlp_forward_one）
// - 方法B：分层核函数前向（src/kernels/mlp_layer.cl -> fc_relu_layer 连续三次）
// 共同点：
// - 启动时一次性加载权重/偏置，创建只读缓冲区
// - 一次初始化OpenCL上下文，推理时仅写入输入并读取输出
// - 支持多次推理复用，避免重复编译/重复分配

#pragma once

#include <CL/cl.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "opencl_utils.hpp"

// 模型权重载体：三层全连接网络（784->128->64->10）
struct ModelWeights {
    std::vector<float> W0, b0, W1, b1, W2, b2;
};

// 从多个候选路径中读取文本文件（第一个能打开的即返回），用于适配不同运行目录
inline std::string read_text_file_first_found(const std::vector<std::string>& candidates) {
    for (const auto& path : candidates) {
        std::ifstream ifs(path);
        if (ifs.is_open()) {
            std::ostringstream oss; oss << ifs.rdbuf();
            return oss.str();
        }
    }
    throw std::runtime_error("Cannot open any of the files");
}

// 读取矩阵权重：首行 "rows cols"，随后按行主序
inline void read_matrix_txt(const std::string &path, int expect_rows, int expect_cols, std::vector<float> &out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("open weight: " + path);
    }
    int rows = 0, cols = 0; f >> rows >> cols;
    if (rows != expect_rows || cols != expect_cols) {
        throw std::runtime_error("shape mismatch: " + path);
    }
    out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            f >> out[static_cast<size_t>(r) * expect_cols + c];
        }
    }
}

// 读取偏置向量：首行 "len 1"，随后元素
inline void read_vector_txt(const std::string &path, int expect_len, std::vector<float> &out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("open bias: " + path);
    }
    int rows = 0, cols = 0; f >> rows >> cols;
    if (rows != expect_len || cols != 1) {
        throw std::runtime_error("shape mismatch: " + path);
    }
    out.resize(expect_len);
    for (int i = 0; i < expect_len; ++i) { f >> out[i]; }
}

// 找到10维logits的最大值索引
inline int argmax10(const float* logits) {
    int best = 0;
    for (int i = 1; i < 10; ++i) { if (logits[i] > logits[best]) best = i; }
    return best;
}

// OpenCL 运行时封装：设备/上下文/队列；生命周期由该结构体管理
struct OpenCLContext {
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    // 初始化：选择设备、创建上下文和命令队列
    void init() {
        device = pick_first_device_print();
        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        checkCLError(err, "create context");
        queue = clCreateCommandQueue(context, device, 0, &err);
        checkCLError(err, "create queue");
    }

    // 析构：按序释放资源
    ~OpenCLContext() {
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

// Method A: Single-kernel forward (mlp_forward.cl -> kernel: mlp_forward_one)
// 方法A：单核函数前向（mlp_forward_one）
class ForwardSingleKernel {
public:
    // 初始化：编译程序/内核、创建常量权重缓冲与复用的输入/输出缓冲
    void initialize(OpenCLContext& clctx, const ModelWeights& weights) {
        ctx_ = &clctx;

        // Load and build program
        std::string src = read_text_file_first_found({
            "src/kernels/A.cl", "../src/kernels/A.cl", "../../src/kernels/A.cl"
        });
        const char* psrc = src.c_str(); size_t slen = src.size();
        cl_int err = CL_SUCCESS;
        prog_ = clCreateProgramWithSource(ctx_->context, 1, &psrc, &slen, &err);
        checkCLError(err, "create program (single)");
        err = clBuildProgram(prog_, 1, &ctx_->device, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::string log = build_log(prog_, ctx_->device);
            std::cerr << "Build log (single):\n" << log << std::endl;
            checkCLError(err, "build program (single)");
        }
        krn_ = clCreateKernel(prog_, "mlp_forward_one", &err);
        checkCLError(err, "create kernel mlp_forward_one");

        // Create constant buffers (weights)
        buf_W0_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 784 * 128, (void*)weights.W0.data(), &err);
        checkCLError(err, "buf_W0");
        buf_b0_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 128, (void*)weights.b0.data(), &err);
        checkCLError(err, "buf_b0");
        buf_W1_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 128 * 64, (void*)weights.W1.data(), &err);
        checkCLError(err, "buf_W1");
        buf_b1_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 64, (void*)weights.b1.data(), &err);
        checkCLError(err, "buf_b1");
        buf_W2_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 64 * 10, (void*)weights.W2.data(), &err);
        checkCLError(err, "buf_W2");
        buf_b2_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 10, (void*)weights.b2.data(), &err);
        checkCLError(err, "buf_b2");

        // Allocate reusable buffers
        buf_x_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY, sizeof(float) * 784, nullptr, &err);
        checkCLError(err, "buf_x");
        buf_out_ = clCreateBuffer(ctx_->context, CL_MEM_WRITE_ONLY, sizeof(float) * 10, nullptr, &err);
        checkCLError(err, "buf_out");
    }

    // 推理：写入 784 输入，调用内核，读取 10 维输出
    void infer(const std::vector<float>& x784, std::array<float,10>& logits_out) {
        if (x784.size() != 784) throw std::runtime_error("input must be length 784");
        cl_int err = CL_SUCCESS;
        err = clEnqueueWriteBuffer(ctx_->queue, buf_x_, CL_TRUE, 0, sizeof(float) * 784, x784.data(), 0, nullptr, nullptr);
        checkCLError(err, "write x");

        int arg = 0;
        err  = clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_x_);
        err |= clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_W0_);
        err |= clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_b0_);
        err |= clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_W1_);
        err |= clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_b1_);
        err |= clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_W2_);
        err |= clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_b2_);
        err |= clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_out_);
        checkCLError(err, "set args single");

        size_t gws = 1;
        checkCLError(clEnqueueNDRangeKernel(ctx_->queue, krn_, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr), "enqueue single");
        checkCLError(clFinish(ctx_->queue), "finish single");

        checkCLError(clEnqueueReadBuffer(ctx_->queue, buf_out_, CL_TRUE, 0, sizeof(float) * 10, logits_out.data(), 0, nullptr, nullptr), "read out single");
    }

    // 资源释放
    ~ForwardSingleKernel() {
        if (buf_x_) clReleaseMemObject(buf_x_);
        if (buf_out_) clReleaseMemObject(buf_out_);
        if (buf_W0_) clReleaseMemObject(buf_W0_);
        if (buf_b0_) clReleaseMemObject(buf_b0_);
        if (buf_W1_) clReleaseMemObject(buf_W1_);
        if (buf_b1_) clReleaseMemObject(buf_b1_);
        if (buf_W2_) clReleaseMemObject(buf_W2_);
        if (buf_b2_) clReleaseMemObject(buf_b2_);
        if (krn_) clReleaseKernel(krn_);
        if (prog_) clReleaseProgram(prog_);
    }

private:
    OpenCLContext* ctx_ = nullptr;
    cl_program prog_ = nullptr;
    cl_kernel krn_ = nullptr;
    cl_mem buf_W0_ = nullptr, buf_b0_ = nullptr, buf_W1_ = nullptr, buf_b1_ = nullptr, buf_W2_ = nullptr, buf_b2_ = nullptr;
    cl_mem buf_x_ = nullptr, buf_out_ = nullptr;
};

// Method B: Layer-wise kernels (kernels/mlp_layer.cl -> kernel: fc_relu_layer used 3 times)
// 方法B：分层核函数前向（fc_relu_layer 连续三次）
class ForwardLayerKernels {
public:
    // 初始化：编译程序/内核、创建常量权重缓冲与复用的中间/输出缓冲
    void initialize(OpenCLContext& clctx, const ModelWeights& weights) {
        ctx_ = &clctx;
        std::string src = read_text_file_first_found({
            "src/kernels/B.cl", "../src/kernels/B.cl", "../../src/kernels/B.cl"
        });
        const char* psrc = src.c_str(); size_t slen = src.size();
        cl_int err = CL_SUCCESS;
        prog_ = clCreateProgramWithSource(ctx_->context, 1, &psrc, &slen, &err);
        checkCLError(err, "create program (layer)");
        err = clBuildProgram(prog_, 1, &ctx_->device, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::string log = build_log(prog_, ctx_->device);
            std::cerr << "Build log (layer):\n" << log << std::endl;
            checkCLError(err, "build program (layer)");
        }
        krn_ = clCreateKernel(prog_, "fc_relu_layer", &err);
        checkCLError(err, "create kernel fc_relu_layer");

        // Constant buffers (weights)
        buf_W0_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 784 * 128, (void*)weights.W0.data(), &err);
        checkCLError(err, "buf_W0 L");
        buf_b0_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 128, (void*)weights.b0.data(), &err);
        checkCLError(err, "buf_b0 L");
        buf_W1_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 128 * 64, (void*)weights.W1.data(), &err);
        checkCLError(err, "buf_W1 L");
        buf_b1_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 64, (void*)weights.b1.data(), &err);
        checkCLError(err, "buf_b1 L");
        buf_W2_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 64 * 10, (void*)weights.W2.data(), &err);
        checkCLError(err, "buf_W2 L");
        buf_b2_ = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * 10, (void*)weights.b2.data(), &err);
        checkCLError(err, "buf_b2 L");

        // Reusable intermediate buffers
        buf_X_  = clCreateBuffer(ctx_->context, CL_MEM_READ_ONLY,  sizeof(float) * 784, nullptr, &err);
        checkCLError(err, "buf_X L");
        buf_H0_ = clCreateBuffer(ctx_->context, CL_MEM_READ_WRITE, sizeof(float) * 128, nullptr, &err);
        checkCLError(err, "buf_H0 L");
        buf_H1_ = clCreateBuffer(ctx_->context, CL_MEM_READ_WRITE, sizeof(float) * 64, nullptr, &err);
        checkCLError(err, "buf_H1 L");
        buf_Y_  = clCreateBuffer(ctx_->context, CL_MEM_READ_WRITE, sizeof(float) * 10, nullptr, &err);
        checkCLError(err, "buf_Y L");
    }

    // 推理：依次执行三层（前两层 ReLU=1，最后一层 ReLU=0），读取 10 维输出
    void infer(const std::vector<float>& x784, std::array<float,10>& logits_out) {
        if (x784.size() != 784) throw std::runtime_error("input must be length 784");
        cl_int err = clEnqueueWriteBuffer(ctx_->queue, buf_X_, CL_TRUE, 0, sizeof(float) * 784, x784.data(), 0, nullptr, nullptr);
        checkCLError(err, "write X L");

        size_t lws[2] = {16, 1};
        auto round_up = [](size_t a, size_t b){ return (a + b - 1) / b * b; };

        // L0: X(784) * W0(784x128) + b0 -> H0(128), relu=1
        {
            const int in0 = 784, out0 = 128, N = 1, apply_relu = 1;
            size_t gws[2] = { round_up((size_t)out0, lws[0]), (size_t)N };
            int arg = 0;
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_X_),  "arg X L0");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_W0_), "arg W0 L0");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_b0_), "arg b0 L0");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_H0_), "arg H0 L0");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &in0),     "arg in0 L0");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &out0),    "arg out0 L0");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &N),       "arg N L0");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &apply_relu), "arg relu L0");
            checkCLError(clEnqueueNDRangeKernel(ctx_->queue, krn_, 2, nullptr, gws, lws, 0, nullptr, nullptr), "enqueue L0");
            checkCLError(clFinish(ctx_->queue), "finish L0");
        }

        // L1: H0(128) * W1(128x64) + b1 -> H1(64), relu=1
        {
            const int in1 = 128, out1 = 64, N = 1, apply_relu = 1;
            size_t gws[2] = { round_up((size_t)out1, lws[0]), (size_t)N };
            int arg = 0;
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_H0_), "arg H0 L1 in");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_W1_), "arg W1 L1");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_b1_), "arg b1 L1");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_H1_), "arg H1 L1 out");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &in1),     "arg in1 L1");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &out1),    "arg out1 L1");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &N),       "arg N L1");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &apply_relu), "arg relu L1");
            checkCLError(clEnqueueNDRangeKernel(ctx_->queue, krn_, 2, nullptr, gws, lws, 0, nullptr, nullptr), "enqueue L1");
            checkCLError(clFinish(ctx_->queue), "finish L1");
        }

        // L2: H1(64) * W2(64x10) + b2 -> Y(10), relu=0
        {
            const int in2 = 64, out2 = 10, N = 1, apply_relu = 0;
            size_t gws[2] = { round_up((size_t)out2, lws[0]), (size_t)N };
            int arg = 0;
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_H1_), "arg H1 L2 in");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_W2_), "arg W2 L2");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_b2_), "arg b2 L2");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(cl_mem), &buf_Y_),  "arg Y L2 out");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &in2),     "arg in2 L2");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &out2),    "arg out2 L2");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &N),       "arg N L2");
            checkCLError(clSetKernelArg(krn_, arg++, sizeof(int),    &apply_relu), "arg relu L2");
            checkCLError(clEnqueueNDRangeKernel(ctx_->queue, krn_, 2, nullptr, gws, lws, 0, nullptr, nullptr), "enqueue L2");
            checkCLError(clFinish(ctx_->queue), "finish L2");
        }

        checkCLError(clEnqueueReadBuffer(ctx_->queue, buf_Y_, CL_TRUE, 0, sizeof(float) * 10, logits_out.data(), 0, nullptr, nullptr), "read Y L");
    }

    // 资源释放
    ~ForwardLayerKernels() {
        if (buf_X_) clReleaseMemObject(buf_X_);
        if (buf_H0_) clReleaseMemObject(buf_H0_);
        if (buf_H1_) clReleaseMemObject(buf_H1_);
        if (buf_Y_) clReleaseMemObject(buf_Y_);
        if (buf_W0_) clReleaseMemObject(buf_W0_);
        if (buf_b0_) clReleaseMemObject(buf_b0_);
        if (buf_W1_) clReleaseMemObject(buf_W1_);
        if (buf_b1_) clReleaseMemObject(buf_b1_);
        if (buf_W2_) clReleaseMemObject(buf_W2_);
        if (buf_b2_) clReleaseMemObject(buf_b2_);
        if (krn_) clReleaseKernel(krn_);
        if (prog_) clReleaseProgram(prog_);
    }

private:
    OpenCLContext* ctx_ = nullptr;
    cl_program prog_ = nullptr;
    cl_kernel krn_ = nullptr;
    cl_mem buf_W0_ = nullptr, buf_b0_ = nullptr, buf_W1_ = nullptr, buf_b1_ = nullptr, buf_W2_ = nullptr, buf_b2_ = nullptr;
    cl_mem buf_X_ = nullptr, buf_H0_ = nullptr, buf_H1_ = nullptr, buf_Y_ = nullptr;
};


