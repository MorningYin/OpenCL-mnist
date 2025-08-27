// ===========================================
// MNIST手写数字识别推理程序 - OpenCL版本
// ===========================================
// 功能说明：
// 1. 读取Python训练好的MLP模型权重文件（.txt格式）
// 2. 使用OpenCL在GPU/CPU上进行前向推理
// 3. 支持单样本推理和批量推理
// 4. 网络结构：784 -> 128 -> 64 -> 10（全连接+ReLU）

// ===========================================
// 标准库头文件包含
// ===========================================
#include <CL/cl.h>        // OpenCL API头文件
#include <fstream>        // 文件流操作（读取权重文件）
#include <iostream>       // 标准输入输出流
#include <sstream>        // 字符串流（解析文件内容）
#include <string>         // 字符串类
#include <vector>         // 动态数组容器
#include <cassert>        // 断言宏（调试用）

// ===========================================
// 辅助函数1：读取文本文件内容
// ===========================================
// 功能：读取OpenCL核函数源代码文件
// 参数：path - 文件路径
// 返回：文件内容的字符串
static std::string read_text_file(const std::string &path) {
    // 创建输入文件流对象
    std::ifstream ifs(path);

    // 检查文件是否成功打开
    if (!ifs.is_open()) {
        // 如果打开失败，抛出运行时错误异常
        throw std::runtime_error("Failed to open file: " + path);
    }

    // 创建输出字符串流，用于存储文件内容
    std::ostringstream oss;

    // 将整个文件内容读取到字符串流中
    // ifs.rdbuf()返回文件流的缓冲区指针
    oss << ifs.rdbuf();

    // 返回文件内容的字符串
    return oss.str();
}

// ===========================================
// 辅助函数2：读取权重矩阵文件
// ===========================================
// 功能：从.txt文件读取权重矩阵（行主序存储）
// 文件格式：第一行是"行数 列数"，后面是矩阵元素（按行存储）
// 参数：
//   path - 权重文件路径
//   expect_rows - 期望的行数
//   expect_cols - 期望的列数
//   out - 输出向量，用于存储读取的矩阵数据
static void read_matrix_txt(const std::string &path, int expect_rows, int expect_cols, std::vector<float> &out) {
    // 创建输入文件流
    std::ifstream f(path);

    // 检查文件是否成功打开
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }

    // 读取矩阵的行数和列数
    int rows = 0, cols = 0;
    f >> rows >> cols;

    // 验证读取到的矩阵尺寸是否与期望一致
    if (rows != expect_rows || cols != expect_cols) {
        throw std::runtime_error("Shape mismatch for " + path +
                                ". Expected: " + std::to_string(expect_rows) + "x" + std::to_string(expect_cols) +
                                ", Got: " + std::to_string(rows) + "x" + std::to_string(cols));
    }

    // 调整输出向量的大小
    // 使用static_cast确保类型转换的安全性
    out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));

    // 按行主序读取矩阵元素
    // 存储格式：out[r * cols + c] 对应矩阵的第r行第c列
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // 读取一个浮点数并存储到对应的位置
            f >> out[static_cast<size_t>(r) * expect_cols + c];
        }
    }
}

// ===========================================
// 辅助函数3：读取偏置向量文件
// ===========================================
// 功能：从.txt文件读取偏置向量
// 文件格式：第一行是"行数 列数"（列数应为1），后面是向量元素
// 参数：
//   path - 偏置文件路径
//   expect_len - 期望的向量长度
//   out - 输出向量，用于存储读取的偏置数据
static void read_vector_txt(const std::string &path, int expect_len, std::vector<float> &out) {
    // 创建输入文件流
    std::ifstream f(path);

    // 检查文件是否成功打开
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open bias file: " + path);
    }

    // 读取向量的行数和列数
    // 偏置向量通常存储为列向量，所以列数应该是1
    int rows = 0, cols = 0;
    f >> rows >> cols;

    // 验证读取到的向量尺寸是否正确
    if (rows != expect_len || cols != 1) {
        throw std::runtime_error("Shape mismatch for " + path +
                                ". Expected: " + std::to_string(expect_len) + "x1" +
                                ", Got: " + std::to_string(rows) + "x" + std::to_string(cols));
    }

    // 调整输出向量的大小
    out.resize(expect_len);

    // 逐个读取向量元素
    for (int i = 0; i < expect_len; ++i) {
        // 读取一个浮点数并存储到输出向量中
        f >> out[i];
    }
}

// ===========================================
// 辅助函数4：选择第一个可用的OpenCL设备
// ===========================================
// 功能：遍历所有OpenCL平台和设备，返回第一个找到的设备
// 优先级：按照平台顺序和设备顺序选择第一个可用设备
// 返回：可用的OpenCL设备ID
static cl_device_id pick_first_device() {
    // ===========================================
    // 获取平台数量
    // ===========================================
    cl_uint numPlatforms = 0;  // 存储平台数量
    // 调用OpenCL API获取平台数量
    // 参数：0表示查询平台数量，nullptr表示不需要平台列表
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);

    // 检查是否有平台可用
    if (err != CL_SUCCESS || numPlatforms == 0) {
        throw std::runtime_error("No OpenCL platforms found. Please install OpenCL drivers.");
    }

    // ===========================================
    // 获取所有平台ID
    // ===========================================
    // 创建平台ID向量，大小为平台数量
    std::vector<cl_platform_id> plats(numPlatforms);

    // 获取所有平台的ID列表
    clGetPlatformIDs(numPlatforms, plats.data(), nullptr);

    // ===========================================
    // 遍历每个平台，寻找可用设备
    // ===========================================
    for (cl_uint p = 0; p < numPlatforms; ++p) {
        // 获取当前平台的设备数量
        cl_uint numDevices = 0;
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);

        // 如果该平台没有设备，继续下一个平台
        if (numDevices == 0) continue;

        // 获取当前平台的所有设备ID
        std::vector<cl_device_id> devs(numDevices);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, numDevices, devs.data(), nullptr);

        // 返回第一个找到的设备
        // 通常是GPU，如果没有GPU则可能是CPU或其他设备
        return devs[0];
    }

    // 如果遍历完所有平台都没有找到设备，抛出错误
    throw std::runtime_error("No OpenCL devices found. Please ensure you have a compatible GPU or CPU.");
}

// ===========================================
// 辅助函数5：找到概率最大的数字类别
// ===========================================
// 功能：在10个类别的logits中找到得分最高的类别索引
// 这相当于在softmax概率中找到最大值对应的类别
// 参数：logits - 长度为10的浮点数向量，表示每个数字(0-9)的得分
// 返回：得分最高的数字类别（0-9）
static int argmax10(const std::vector<float>& logits) {
    // 初始化最佳类别为第0类（数字0）
    int best = 0;

    // 遍历第1到第9类（数字1-9）
    for (int i = 1; i < 10; ++i) {
        // 如果当前类别的得分高于当前最佳类别
        if (logits[i] > logits[best]) {
            // 更新最佳类别索引
            best = i;
        }
    }

    // 返回得分最高的类别索引
    return best;
}

// ===========================================
// 主函数：MNIST数字识别推理程序入口
// ===========================================
// 命令行参数：
//   argv[0] - 程序名
//   argv[1] - 可选：包含784个像素值的输入文件路径
//
// 程序流程：
// 1. 读取模型权重文件
// 2. 初始化OpenCL环境
// 3. 编译并执行核函数
// 4. 输出识别结果
int main(int argc, char** argv) {
    try {
        // ===========================================
        // 阶段1：读取模型权重和偏置
        // ===========================================

        // 定义权重文件所在目录
        const std::string root = "src/model_out/";

        // 定义网络各层的尺寸（基于model_meta.txt）
        // linear_0: 784输入 -> 128输出
        const int in0 = 784, out0 = 128;
        // linear_1: 128输入 -> 64输出
        const int in1 = 128, out1 = 64;
        // linear_2: 64输入 -> 10输出（10个数字类别）
        const int in2 = 64,  out2 = 10;

        // 声明存储权重和偏置的向量
        std::vector<float> W0, b0, W1, b1, W2, b2;

        // 读取第一层权重矩阵：784x128
        read_matrix_txt(root + "linear0_W.txt", in0, out0, W0);
        // 读取第一层偏置向量：128
        read_vector_txt(root + "linear0_b.txt", out0, b0);
        // 读取第二层权重矩阵：128x64
        read_matrix_txt(root + "linear1_W.txt", in1, out1, W1);
        // 读取第二层偏置向量：64
        read_vector_txt(root + "linear1_b.txt", out1, b1);
        // 读取第三层权重矩阵：64x10
        read_matrix_txt(root + "linear2_W.txt", in2, out2, W2);
        // 读取第三层偏置向量：10
        read_vector_txt(root + "linear2_b.txt", out2, b2);

        // ===========================================
        // 阶段2：准备输入数据
        // ===========================================

        // 准备输入数据：创建784维的浮点数向量
        // MNIST图像是28x28=784像素，我们将其展平为一维向量
        std::vector<float> x(784, 0.0f);  // 初始化为全0向量

        // 如果命令行提供了第二个参数（输入文件路径）
        if (argc == 2) {
            // 从文本文件读取784个浮点数作为输入
            // 文件格式：每行一个浮点数，共784行
            std::ifstream fx(argv[1]);
            if (fx.is_open()) {
                // 逐个读取像素值，覆盖默认的全0输入
                for (int i = 0; i < 784 && fx; ++i) {
                    fx >> x[i];
                }
            }
        }

        // ===========================================
        // 阶段3：初始化OpenCL环境
        // ===========================================

        // 选择第一个可用的OpenCL设备（通常是GPU）
        cl_device_id dev = pick_first_device();

        // 定义OpenCL错误码变量
        cl_int err = CL_SUCCESS;

        // 创建OpenCL上下文（context）
        // 上下文是OpenCL程序运行的环境，包含设备、内存等资源
        cl_context ctx = clCreateContext(nullptr,     // 属性列表（使用默认）
                                        1,           // 设备数量
                                        &dev,        // 设备列表
                                        nullptr,     // 错误回调函数
                                        nullptr,     // 用户数据
                                        &err);       // 错误码输出
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL context. Error code: " + std::to_string(err));
        }

        // 创建命令队列（command queue）
        // 命令队列用于向设备提交计算任务和内存操作
        cl_command_queue q = clCreateCommandQueue(ctx,     // 上下文
                                                 dev,     // 目标设备
                                                 0,       // 队列属性（使用默认）
                                                 &err);   // 错误码输出
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create command queue. Error code: " + std::to_string(err));
        }

        // ===========================================
        // 阶段4：编译OpenCL核函数
        // ===========================================

        // 读取核函数源代码文件（单核前向 A.cl，导出 mlp_forward_one）
        std::string kernel_src = read_text_file("src/kernels/A.cl");

        // 将字符串转换为C风格的字符指针（OpenCL API要求）
        const char* src = kernel_src.c_str();
        size_t len = kernel_src.size();

        // 创建OpenCL程序对象
        // 程序对象包含编译后的核函数代码
        cl_program prog = clCreateProgramWithSource(ctx,      // 上下文
                                                   1,        // 源代码字符串数量
                                                   &src,     // 源代码字符串指针数组
                                                   &len,     // 每个字符串的长度
                                                   &err);    // 错误码输出
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create program from source. Error code: " + std::to_string(err));
        }

        // 编译程序
        // 这个步骤将源代码编译为设备特定的二进制代码
        err = clBuildProgram(prog,        // 程序对象
                            1,           // 设备数量
                            &dev,        // 目标设备
                            "",          // 编译选项（使用默认）
                            nullptr,     // 编译回调函数
                            nullptr);    // 用户数据

        // 如果编译失败，获取并显示编译日志
        if (err != CL_SUCCESS) {
            // 获取编译日志的大小
            size_t logSize = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

            // 分配缓冲区存储编译日志
            std::string log(logSize, '\0');

            // 获取编译日志内容
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);

            // 输出编译错误信息
            std::cerr << "OpenCL compilation failed with error code: " << err << std::endl;
            std::cerr << "Build log:\n" << log << std::endl;
            throw std::runtime_error("Failed to build OpenCL program. Check build log above.");
        }

        // ===========================================
        // 阶段5：创建并设置核函数
        // ===========================================

        // 创建核函数对象
        // "mlp_forward_one"是我们在.cl文件中定义的核函数名
        cl_kernel krn = clCreateKernel(prog,              // 程序对象
                                      "mlp_forward_one", // 核函数名
                                      &err);             // 错误码输出
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create kernel 'mlp_forward_one'. Error code: " + std::to_string(err));
        }

        // ===========================================
        // 阶段6：创建设备内存缓冲区
        // ===========================================

        // 创建输入数据缓冲区（只读）
        // 将主机内存中的输入向量x复制到设备内存
        cl_mem buf_x = clCreateBuffer(ctx,                          // 上下文
                                     CL_MEM_READ_ONLY |            // 内存标志：只读
                                     CL_MEM_COPY_HOST_PTR,         // 复制主机数据到设备
                                     sizeof(float) * 784,          // 缓冲区大小（字节）
                                     x.data(),                     // 主机数据指针
                                     &err);                        // 错误码输出

        // 创建权重矩阵缓冲区（只读）
        cl_mem buf_W0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * in0 * out0, W0.data(), &err);  // 第一层权重：784x128
        cl_mem buf_b0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * out0, b0.data(), &err);       // 第一层偏置：128

        cl_mem buf_W1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * in1 * out1, W1.data(), &err);  // 第二层权重：128x64
        cl_mem buf_b1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * out1, b1.data(), &err);       // 第二层偏置：64

        cl_mem buf_W2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * in2 * out2, W2.data(), &err);  // 第三层权重：64x10
        cl_mem buf_b2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * out2, b2.data(), &err);       // 第三层偏置：10

        // 创建输出缓冲区（只写）
        // 初始化为0，不复制主机数据
        std::vector<float> logits(out2, 0.0f);  // 主机端的输出向量
        cl_mem buf_out = clCreateBuffer(ctx,                          // 上下文
                                       CL_MEM_WRITE_ONLY,             // 内存标志：只写
                                       sizeof(float) * out2,          // 缓冲区大小：10个float
                                       nullptr,                       // 不提供初始数据
                                       &err);                         // 错误码输出

        // ===========================================
        // 阶段7：设置核函数参数
        // ===========================================

        // 为核函数设置参数（按照.cl文件中定义的顺序）
        int arg_idx = 0;  // 参数索引跟踪

        // 参数0：输入数据缓冲区
        arg_idx = 0;
        err = clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_x);

        // 参数1-7：权重和偏置缓冲区
        arg_idx = 1; err |= clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_W0);  // 第一层权重
        arg_idx = 2; err |= clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_b0);  // 第一层偏置
        arg_idx = 3; err |= clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_W1);  // 第二层权重
        arg_idx = 4; err |= clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_b1);  // 第二层偏置
        arg_idx = 5; err |= clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_W2);  // 第三层权重
        arg_idx = 6; err |= clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_b2);  // 第三层偏置
        arg_idx = 7; err |= clSetKernelArg(krn, arg_idx, sizeof(cl_mem), &buf_out); // 输出缓冲区

        // 检查参数设置是否成功
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to set kernel arguments at index " + std::to_string(arg_idx) +
                                   ". Error code: " + std::to_string(err));
        }

        // ===========================================
        // 阶段8：执行核函数
        // ===========================================

        // 设置全局工作组大小（work-group dimensions）
        // 我们只有一个work-item（单样本推理）
        size_t global_work_size = 1;

        // 提交核函数执行到命令队列
        err = clEnqueueNDRangeKernel(q,                    // 命令队列
                                    krn,                  // 核函数
                                    1,                    // 工作维度（1D）
                                    nullptr,              // 全局工作偏移（使用默认）
                                    &global_work_size,    // 全局工作组大小
                                    nullptr,              // 局部工作组大小（让OpenCL决定）
                                    0,                    // 等待事件数量
                                    nullptr,              // 等待事件列表
                                    nullptr);             // 完成事件

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue kernel execution. Error code: " + std::to_string(err));
        }

        // 等待命令队列中的所有命令完成
        clFinish(q);

        // ===========================================
        // 阶段9：读取计算结果
        // ===========================================

        // 将设备内存中的输出数据复制回主机内存
        err = clEnqueueReadBuffer(q,                      // 命令队列
                                 buf_out,                 // 源缓冲区（设备）
                                 CL_TRUE,                 // 阻塞读取（等待完成）
                                 0,                      // 缓冲区偏移
                                 sizeof(float) * out2,   // 读取字节数
                                 logits.data(),          // 目标内存（主机）
                                 0,                      // 等待事件数量
                                 nullptr,                // 等待事件列表
                                 nullptr);               // 完成事件

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read output buffer. Error code: " + std::to_string(err));
        }

        // ===========================================
        // 阶段10：处理和显示结果
        // ===========================================

        // 使用argmax找到得分最高的类别
        int predicted_class = argmax10(logits);

        // 输出识别结果
        std::cout << "Predicted class: " << predicted_class << std::endl;
        std::cout << "Logits:";

        // 显示所有10个类别的得分
        for (int i = 0; i < out2; ++i) {
            std::cout << " " << logits[i];
        }
        std::cout << std::endl;

        // ===========================================
        // 阶段11：清理资源
        // ===========================================
        // OpenCL资源管理非常重要，需要手动释放所有创建的对象
        // 防止内存泄漏

        // 释放内存缓冲区
        clReleaseMemObject(buf_x);   // 释放输入数据缓冲区
        clReleaseMemObject(buf_W0);  // 释放第一层权重缓冲区
        clReleaseMemObject(buf_b0);  // 释放第一层偏置缓冲区
        clReleaseMemObject(buf_W1);  // 释放第二层权重缓冲区
        clReleaseMemObject(buf_b1);  // 释放第二层偏置缓冲区
        clReleaseMemObject(buf_W2);  // 释放第三层权重缓冲区
        clReleaseMemObject(buf_b2);  // 释放第三层偏置缓冲区
        clReleaseMemObject(buf_out); // 释放输出缓冲区

        // 释放核函数对象
        clReleaseKernel(krn);

        // 释放程序对象
        clReleaseProgram(prog);

        // 释放命令队列
        clReleaseCommandQueue(q);

        // 释放上下文（最后释放，因为其他对象都依赖于上下文）
        clReleaseContext(ctx);

        // 程序执行成功，返回0
        return 0;

    } catch (const std::exception &ex) {
        // ===========================================
        // 异常处理
        // ===========================================
        // 捕获并处理程序运行中的所有异常
        // 输出错误信息到标准错误流
        std::cerr << "Error: " << ex.what() << std::endl;

        // 返回非零值表示程序异常终止
        return 1;
    }
}


