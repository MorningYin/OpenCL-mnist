// ===========================================
// MNIST手写数字识别推理程序 - OpenCL批量版本
// ===========================================
// 功能说明：
// 1. 读取Python训练好的MLP模型权重文件（.txt格式）
// 2. 使用OpenCL在GPU/CPU上进行批量前向推理
// 3. 支持单样本和批量推理，支持命令行参数控制
// 4. 网络结构：784 -> 128 -> 64 -> 10（全连接+ReLU）
// 5. 实现分块并行计算，提升GPU利用率

// ===========================================
// 标准库头文件包含
// ===========================================
#include <CL/cl.h>        // OpenCL API头文件，提供所有OpenCL函数和常量
#include <algorithm>      // 算法库，提供std::max等函数
#include <fstream>        // 文件流操作，用于读取权重文件和样本文件
#include <iostream>       // 标准输入输出流，用于控制台输出
#include <numeric>        // 数值算法库，提供数学运算函数
#include <sstream>        // 字符串流，用于字符串处理
#include <string>         // 字符串类，提供字符串操作
#include <vector>         // 动态数组容器，用于存储权重、偏置、输入输出数据

// ===========================================
// 项目自定义头文件
// ===========================================
#include "opencl_utils.hpp"  // OpenCL工具函数，包含错误检查、设备选择、构建日志等

// ===========================================
// 辅助函数1：读取文本文件内容
// ===========================================
// 功能：读取OpenCL核函数源代码文件到字符串中
// 参数：path - 文件路径
// 返回：文件内容的字符串
// 用途：读取.cl内核文件，用于OpenCL程序编译
static std::string read_text_file(const std::string &path) {
    // 创建输入文件流对象，打开指定路径的文件
    std::ifstream ifs(path);
    
    // 检查文件是否成功打开
    if (!ifs.is_open()) {
        // 如果打开失败，抛出运行时错误异常，包含文件路径信息
        throw std::runtime_error("open file: " + path);
    }
    
    // 创建输出字符串流，用于存储文件内容
    std::ostringstream oss;
    
    // 将整个文件内容读取到字符串流中
    // ifs.rdbuf()返回文件流的缓冲区指针，<<操作符将内容写入字符串流
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
//   expect_rows - 期望的行数（输入维度）
//   expect_cols - 期望的列数（输出维度）
//   out - 输出向量，用于存储读取的矩阵数据
// 用途：读取神经网络权重矩阵，如W0(784x128)、W1(128x64)、W2(64x10)
static void read_matrix_txt(const std::string &path, int expect_rows, int expect_cols, std::vector<float> &out) {
    // 创建输入文件流，打开权重文件
    std::ifstream f(path);
    
    // 检查文件是否成功打开
    if (!f.is_open()) {
        // 如果打开失败，抛出运行时错误异常
        throw std::runtime_error("open weight: " + path);
    }
    
    // 读取矩阵的行数和列数
    // 文件第一行格式：rows cols（例如：784 128）
    int rows = 0, cols = 0;
    f >> rows >> cols;  // 从文件流中读取两个整数
    
    // 验证读取到的矩阵尺寸是否与期望一致
    if (rows != expect_rows || cols != expect_cols) {
        // 如果尺寸不匹配，抛出运行时错误异常
        throw std::runtime_error("shape mismatch: " + path);
    }
    
    // 调整输出向量的大小
    // 使用static_cast确保类型转换的安全性，避免数据丢失
    // 矩阵总元素数 = 行数 × 列数
    out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    
    // 按行主序读取矩阵元素
    // 存储格式：out[r * cols + c] 对应矩阵的第r行第c列
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // 读取一个浮点数并存储到对应的位置
            // 行主序：先存储第一行的所有元素，再存储第二行的所有元素...
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
//   expect_len - 期望的向量长度（输出维度）
//   out - 输出向量，用于存储读取的偏置数据
// 用途：读取神经网络偏置向量，如b0(128)、b1(64)、b2(10)
static void read_vector_txt(const std::string &path, int expect_len, std::vector<float> &out) {
    // 创建输入文件流，打开偏置文件
    std::ifstream f(path);
    
    // 检查文件是否成功打开
    if (!f.is_open()) {
        // 如果打开失败，抛出运行时错误异常
        throw std::runtime_error("open bias: " + path);
    }
    
    // 读取向量的行数和列数
    // 偏置向量通常存储为列向量，所以列数应该是1
    // 文件第一行格式：rows cols（例如：128 1）
    int rows = 0, cols = 0;
    f >> rows >> cols;
    
    // 验证读取到的向量尺寸是否正确
    if (rows != expect_len || cols != 1) {
        // 如果尺寸不匹配，抛出运行时错误异常
        throw std::runtime_error("shape mismatch: " + path);
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
// 辅助函数4：找到概率最大的数字类别
// ===========================================
// 功能：在10个类别的logits中找到得分最高的类别索引
// 这相当于在softmax概率中找到最大值对应的类别
// 参数：logits - 长度为10的浮点数向量，表示每个数字(0-9)的得分
// 返回：得分最高的数字类别（0-9）
// 用途：将神经网络的输出logits转换为最终的预测类别
static int argmax10(const float* logits) {
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
//   --batch N - 可选：指定批量大小N
//
// 程序流程：
// 1. 解析命令行参数，确定批量大小
// 2. 读取模型权重和偏置文件
// 3. 准备输入数据（单样本或批量）
// 4. 初始化OpenCL环境
// 5. 编译OpenCL核函数
// 6. 创建设备内存缓冲区
// 7. 依次执行三层神经网络推理
// 8. 读取结果并输出预测
// 9. 清理OpenCL资源
int main(int argc, char** argv) {
    try {
        // ===========================================
        // 阶段1：定义网络架构参数
        // ===========================================
        // 这些常量定义了三层全连接网络的输入输出维度
        // 与Python训练时的网络结构保持一致
        
        // 第一层：输入784维（28x28图像展平） -> 输出128维
        const int in0 = 784, out0 = 128;
        
        // 第二层：输入128维 -> 输出64维
        const int in1 = 128, out1 = 64;
        
        // 第三层：输入64维 -> 输出10维（10个数字类别）
        const int in2 = 64,  out2 = 10;

        // ===========================================
        // 阶段2：解析命令行参数
        // ===========================================
        // 支持两种命令行格式：
        // 1. ./mnist_batch.exe --batch 32          （批量推理，32个样本）
        // 2. ./mnist_batch.exe sample.txt          （单样本推理，从文件读取）
        // 3. ./mnist_batch.exe                     （单样本推理，使用全0输入）
        
        // 默认批量大小为1（单样本推理）
        int N = 1;
        
        // 遍历命令行参数，查找--batch选项
        for (int i = 1; i + 1 < argc; ++i) {
            // 检查当前参数是否为--batch
            if (std::string(argv[i]) == "--batch") {
                // 下一个参数应该是批量大小
                // std::atoi将字符串转换为整数，std::max确保N至少为1
                N = std::max(1, std::atoi(argv[i + 1]));
            }
        }

        // ===========================================
        // 阶段3：读取模型权重和偏置
        // ===========================================
        // 从src/model_out/目录读取Python训练好的权重文件
        // 这些文件是Python脚本生成的，包含训练好的网络参数
        
        // 定义权重文件所在目录
        const std::string root = "src/model_out/";
        
        // 声明存储权重和偏置的向量
        // 每个向量将存储对应层的参数
        std::vector<float> W0, b0, W1, b1, W2, b2;
        
        // 读取第一层权重矩阵：784x128
        // 文件格式：第一行"784 128"，后面是784*128个浮点数
        read_matrix_txt(root + "linear0_W.txt", in0, out0, W0);
        
        // 读取第一层偏置向量：128
        // 文件格式：第一行"128 1"，后面是128个浮点数
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
        // 阶段4：准备输入数据
        // ===========================================
        // 创建输入数据向量，支持批量处理
        // 数据布局：X[b*784 + i] 表示第b个样本的第i个像素值
        
        // 创建输入数据向量，大小为 N * 784
        // 初始化为全0，这样如果没有提供输入文件，程序仍能正常运行
        std::vector<float> X(static_cast<size_t>(N) * in0, 0.0f);
        
        // 如果用户提供了样本文件作为第一个参数（且不是--batch）
        if (argc >= 2 && std::string(argv[1]) != "--batch") {
            // 尝试打开指定的文件
            std::ifstream fx(argv[1]);
            if (fx.is_open()) {
                // 读取第一个样本的像素值（784个浮点数）
                // 文件格式：每行一个像素值，共784行
                for (int i = 0; i < in0 && fx; ++i) {
                    fx >> X[i];  // 读取像素值到第一个样本位置
                }
            }
            // 注意：这里只读取第一个样本，其他样本保持为0
            // 如果需要读取多个样本，可以扩展这个逻辑
        }

        // ===========================================
        // 阶段5：初始化OpenCL环境
        // ===========================================
        // 设置OpenCL计算环境，包括设备选择、上下文创建、命令队列等
        // 这些是OpenCL程序运行的基础设施
        
        // 选择第一个可用的OpenCL设备（通常是GPU）
        // pick_first_device_print()函数会遍历所有平台和设备，选择第一个可用的
        // 并打印设备信息到控制台
        cl_device_id dev = pick_first_device_print();
        
        // 定义OpenCL错误码变量
        // 用于存储OpenCL API调用的返回状态
        cl_int err = CL_SUCCESS;
        
        // 创建OpenCL上下文（context）
        // 上下文是OpenCL程序运行的环境，包含设备、内存等资源
        // 参数说明：
        //   nullptr - 属性列表（使用默认）
        //   1 - 设备数量
        //   &dev - 设备列表指针
        //   nullptr - 错误回调函数（使用默认）
        //   nullptr - 用户数据（不使用）
        //   &err - 错误码输出
        cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
        checkCLError(err, "create context");  // 检查是否创建成功
        
        // 创建命令队列（command queue）
        // 命令队列用于向设备提交计算任务和内存操作
        // 参数说明：
        //   ctx - 上下文
        //   dev - 目标设备
        //   0 - 队列属性（使用默认，顺序执行）
        //   &err - 错误码输出
        cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
        checkCLError(err, "create queue");  // 检查是否创建成功

        // ===========================================
        // 阶段6：编译OpenCL核函数
        // ===========================================
        // 将OpenCL内核源代码编译为设备特定的二进制代码
        // 这是OpenCL程序的核心部分
        
        // 读取内核源代码文件
        // 内核改名为 B.cl（fc_relu_layer）
        std::string src = read_text_file("src/kernels/B.cl");
        
        // 将C++字符串转换为C风格的字符指针（OpenCL API要求）
        const char* psrc = src.c_str();  // 获取字符串的C风格指针
        size_t slen = src.size();        // 获取字符串长度
        
        // 创建OpenCL程序对象
        // 程序对象包含编译后的核函数代码
        // 参数说明：
        //   ctx - 上下文
        //   1 - 源代码字符串数量
        //   &psrc - 源代码字符串指针数组
        //   &slen - 每个字符串的长度
        //   &err - 错误码输出
        cl_program prog = clCreateProgramWithSource(ctx, 1, &psrc, &slen, &err);
        checkCLError(err, "create program");  // 检查是否创建成功
        
        // 编译程序
        // 这个步骤将源代码编译为设备特定的二进制代码
        // 编译选项：可以根据设备厂商添加优化选项
        std::string opts = "";  // 空字符串表示使用默认编译选项
        // 可以添加的优化选项：
        //   "-cl-fast-relaxed-math" - 允许数学运算的快速近似（提升性能）
        //   "-cl-mad-enable" - 启用乘加融合指令
        //   "-cl-no-signed-zeros" - 忽略符号零
        
        // 执行编译
        err = clBuildProgram(prog,        // 程序对象
                            1,           // 设备数量
                            &dev,        // 目标设备
                            opts.c_str(), // 编译选项
                            nullptr,     // 编译回调函数
                            nullptr);    // 用户数据
        
        // 如果编译失败，获取并显示编译日志
        if (err != CL_SUCCESS) {
            // 输出编译错误信息
            std::cerr << "Build log:\n" << build_log(prog, dev) << std::endl;
            // 抛出异常，终止程序
            checkCLError(err, "build program");
        }

        // 创建核函数对象
        // "fc_relu_layer"是我们在.cl文件中定义的核函数名
        // 这个核函数实现了全连接层+可选ReLU激活的计算
        cl_kernel krn = clCreateKernel(prog,              // 程序对象
                                      "fc_relu_layer",    // 核函数名
                                      &err);              // 错误码输出
        checkCLError(err, "create kernel fc_relu_layer");  // 检查是否创建成功

        // ===========================================
        // 阶段7：创建设备内存缓冲区
        // ===========================================
        // 在GPU/CPU设备上分配内存，用于存储输入数据、权重、偏置和中间结果
        // 这些缓冲区是主机内存和设备内存之间的桥梁
        
        // 创建输入数据缓冲区（只读）
        // 将主机内存中的输入向量X复制到设备内存
        cl_mem buf_X = clCreateBuffer(ctx,                          // 上下文
                                     CL_MEM_READ_ONLY |            // 内存标志：只读
                                     CL_MEM_COPY_HOST_PTR,         // 复制主机数据到设备
                                     sizeof(float) * X.size(),     // 缓冲区大小（字节）
                                     X.data(),                     // 主机数据指针
                                     &err);                        // 错误码输出
        checkCLError(err, "buf_X");  // 检查是否创建成功
        
        // 创建权重矩阵缓冲区（只读）
        // 这些缓冲区存储训练好的网络参数，在推理过程中不会改变
        cl_mem buf_W0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * W0.size(), W0.data(), &err);  // 第一层权重：784x128
        cl_mem buf_b0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * b0.size(), b0.data(), &err);  // 第一层偏置：128

        cl_mem buf_W1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * W1.size(), W1.data(), &err);  // 第二层权重：128x64
        cl_mem buf_b1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * b1.size(), b1.data(), &err);  // 第二层偏置：64

        cl_mem buf_W2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * W2.size(), W2.data(), &err);  // 第三层权重：64x10
        cl_mem buf_b2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(float) * b2.size(), b2.data(), &err);  // 第三层偏置：10
        checkCLError(err, "buf_W2/b2");  // 检查最后两个缓冲区是否创建成功

        // 创建中间结果和输出缓冲区（读写）
        // 这些缓冲区用于存储各层的输出结果
        cl_mem buf_H0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)N * out0, nullptr, &err);  // 第一层输出：Nx128
        cl_mem buf_H1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)N * out1, nullptr, &err);  // 第二层输出：Nx64
        cl_mem buf_YP = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * (size_t)N * out2, nullptr, &err);  // 最终输出：Nx10
        checkCLError(err, "buf_H*/buf_YP");  // 检查中间缓冲区是否创建成功

        // ===========================================
        // 阶段8：配置NDRange参数
        // ===========================================
        // NDRange是OpenCL的并行执行模型，定义了工作项的分布方式
        // 我们使用2D NDRange：第一维是输出维度，第二维是批量大小
        
        // 设置局部工作组大小（local work-group size）
        // local[0] = 16：每个工作组在输出维度上有16个工作项
        // local[1] = 1：每个工作组在批量维度上有1个工作项
        size_t lws[2] = {16, 1};
        
        // 定义向上取整函数
        // 用于确保全局工作组大小是局部工作组大小的整数倍
        // 这是OpenCL的要求，否则会报错
        auto round_up = [](size_t a, size_t b){ 
            return (a + b - 1) / b * b;  // 数学公式：(a + b - 1) / b * b
        };

        // ===========================================
        // 阶段9：执行第一层推理（X -> H0）
        // ===========================================
        // 第一层：输入784维 -> 输出128维，应用ReLU激活函数
        // 计算：H0 = ReLU(X * W0 + b0)
        {
            // 设置是否应用ReLU激活函数（1表示应用，0表示不应用）
            const int apply_relu = 1;
            
            // 计算全局工作组大小（global work-group size）
            // gws[0]：输出维度，向上取整到16的倍数（128 -> 128）
            // gws[1]：批量大小N
            size_t gws[2] = { round_up((size_t)out0, lws[0]), (size_t)N };
            
            // 设置核函数参数
            // 参数顺序必须与.cl文件中定义的完全一致
            int arg = 0;  // 参数索引，从0开始
            
            // 参数0：输入数据缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_X),  "arg X");
            // 参数1：第一层权重矩阵缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_W0), "arg W0");
            // 参数2：第一层偏置向量缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_b0), "arg b0");
            // 参数3：第一层输出缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_H0), "arg H0");
            // 参数4：输入维度
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &in0),    "arg in0");
            // 参数5：输出维度
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &out0),   "arg out0");
            // 参数6：批量大小
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &N),      "arg N");
            // 参数7：是否应用ReLU
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &apply_relu), "arg relu");
            
            // 提交核函数执行到命令队列
            // 使用2D NDRange，全局大小gws，局部大小lws
            checkCLError(clEnqueueNDRangeKernel(q, krn, 2, nullptr, gws, lws, 0, nullptr, nullptr), "enqueue L0");
            
            // 等待第一层计算完成
            // 这是层间同步的关键：确保第一层完全计算完成后，才能开始第二层
            checkCLError(clFinish(q), "finish L0");
        }

        // ===========================================
        // 阶段10：执行第二层推理（H0 -> H1）
        // ===========================================
        // 第二层：输入128维 -> 输出64维，应用ReLU激活函数
        // 计算：H1 = ReLU(H0 * W1 + b1)
        // 注意：这里使用第一层的输出H0作为输入
        {
            // 设置是否应用ReLU激活函数（1表示应用）
            const int apply_relu = 1;
            
            // 计算全局工作组大小
            // gws[0]：输出维度，向上取整到16的倍数（64 -> 64）
            // gws[1]：批量大小N
            size_t gws[2] = { round_up((size_t)out1, lws[0]), (size_t)N };
            
            // 重新设置核函数参数（因为每次调用都需要重新设置）
            int arg = 0;  // 参数索引，从0开始
            
            // 参数0：输入数据缓冲区（现在是第一层的输出H0）
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_H0), "arg H0 in");
            // 参数1：第二层权重矩阵缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_W1), "arg W1");
            // 参数2：第二层偏置向量缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_b1), "arg b1");
            // 参数3：第二层输出缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_H1), "arg H1 out");
            // 参数4：输入维度（128）
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &in1),    "arg in1");
            // 参数5：输出维度（64）
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &out1),   "arg out1");
            // 参数6：批量大小
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &N),      "arg N1");
            // 参数7：是否应用ReLU
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &apply_relu), "arg relu1");
            
            // 提交核函数执行到命令队列
            checkCLError(clEnqueueNDRangeKernel(q, krn, 2, nullptr, gws, lws, 0, nullptr, nullptr), "enqueue L1");
            
            // 等待第二层计算完成
            // 确保第二层完全计算完成后，才能开始第三层
            checkCLError(clFinish(q), "finish L1");
        }

        // ===========================================
        // 阶段11：执行第三层推理（H1 -> Y）
        // ===========================================
        // 第三层：输入64维 -> 输出10维，不应用ReLU激活函数
        // 计算：Y = H1 * W2 + b2（直接输出logits，不经过激活函数）
        // 这是输出层，通常不应用激活函数，直接输出原始得分
        {
            // 设置是否应用ReLU激活函数（0表示不应用）
            const int apply_relu = 0;
            
            // 计算全局工作组大小
            // gws[0]：输出维度，向上取整到16的倍数（10 -> 16）
            // gws[1]：批量大小N
            size_t gws[2] = { round_up((size_t)out2, lws[0]), (size_t)N };
            
            // 重新设置核函数参数
            int arg = 0;  // 参数索引，从0开始
            
            // 参数0：输入数据缓冲区（现在是第二层的输出H1）
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_H1), "arg H1 in");
            // 参数1：第三层权重矩阵缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_W2), "arg W2");
            // 参数2：第三层偏置向量缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_b2), "arg b2");
            // 参数3：最终输出缓冲区
            checkCLError(clSetKernelArg(krn, arg++, sizeof(cl_mem), &buf_YP), "arg Y out");
            // 参数4：输入维度（64）
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &in2),    "arg in2");
            // 参数5：输出维度（10）
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &out2),   "arg out2");
            // 参数6：批量大小
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &N),      "arg N2");
            // 参数7：是否应用ReLU
            checkCLError(clSetKernelArg(krn, arg++, sizeof(int),    &apply_relu), "arg relu2");
            
            // 提交核函数执行到命令队列
            checkCLError(clEnqueueNDRangeKernel(q, krn, 2, nullptr, gws, lws, 0, nullptr, nullptr), "enqueue L2");
            
            // 等待第三层计算完成
            // 确保所有计算都完成后，才能读取结果
            checkCLError(clFinish(q), "finish L2");
        }

        // ===========================================
        // 阶段12：读取计算结果并输出预测
        // ===========================================
        // 将设备内存中的计算结果复制回主机内存，并进行后处理
        
        // 创建主机端向量存储logits结果
        // 大小为 N * 10，每个样本有10个类别的得分
        std::vector<float> logits(static_cast<size_t>(N) * out2, 0.0f);
        
        // 从设备内存读取最终输出结果
        // 参数说明：
        //   q - 命令队列
        //   buf_YP - 源缓冲区（设备端）
        //   CL_TRUE - 阻塞读取（等待完成）
        //   0 - 缓冲区偏移
        //   sizeof(float) * logits.size() - 读取字节数
        //   logits.data() - 目标内存（主机端）
        //   0 - 等待事件数量
        //   nullptr - 等待事件列表
        //   nullptr - 完成事件
        checkCLError(clEnqueueReadBuffer(q, buf_YP, CL_TRUE, 0, sizeof(float) * logits.size(), logits.data(), 0, nullptr, nullptr), "read logits");

        // 处理每个样本的预测结果
        for (int b = 0; b < N; ++b) {
            // 获取当前样本的logits行指针
            // logits按行主序存储：logits[b*10 + j] 表示第b个样本第j个类别的得分
            const float* row = logits.data() + (size_t)b * out2;
            
            // 使用argmax找到得分最高的类别
            int pred = argmax10(row);
            
            // 输出预测结果
            std::cout << "Sample " << b << " Predicted: " << pred << "\nLogits:";
            
            // 输出所有10个类别的得分（用于调试和分析）
            for (int j = 0; j < out2; ++j) {
                std::cout << ' ' << row[j];
            }
            std::cout << '\n';
        }

        // ===========================================
        // 阶段13：清理OpenCL资源
        // ===========================================
        // OpenCL资源管理非常重要，需要手动释放所有创建的对象
        // 防止内存泄漏和资源占用
        
        // 释放内存缓冲区
        clReleaseMemObject(buf_X);   // 释放输入数据缓冲区
        clReleaseMemObject(buf_W0);  // 释放第一层权重缓冲区
        clReleaseMemObject(buf_b0);  // 释放第一层偏置缓冲区
        clReleaseMemObject(buf_W1);  // 释放第二层权重缓冲区
        clReleaseMemObject(buf_b1);  // 释放第二层偏置缓冲区
        clReleaseMemObject(buf_W2);  // 释放第三层权重缓冲区
        clReleaseMemObject(buf_b2);  // 释放第三层偏置缓冲区
        clReleaseMemObject(buf_H0);  // 释放第一层输出缓冲区
        clReleaseMemObject(buf_H1);  // 释放第二层输出缓冲区
        clReleaseMemObject(buf_YP);  // 释放最终输出缓冲区

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
        
    } catch (const std::exception& ex) {
        // ===========================================
        // 异常处理
        // ===========================================
        // 捕获并处理程序运行中的所有异常
        // 包括文件读取错误、OpenCL错误、内存分配错误等
        
        // 输出错误信息到标准错误流
        std::cerr << "Error: " << ex.what() << std::endl;
        
        // 返回非零值表示程序异常终止
        return 1;
    }
}
