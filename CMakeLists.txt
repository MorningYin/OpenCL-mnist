# ===========================================
# CMake配置文件：MNIST手写数字识别OpenCL项目
# ===========================================
# 该文件定义了项目的构建规则和依赖关系

# ===========================================
# 基本项目配置
# ===========================================

# 指定CMake最低版本要求
# VERSION 3.20 是支持现代CMake特性的稳定版本
cmake_minimum_required(VERSION 3.20)

# 定义项目名称和使用的编程语言
# 项目名：EnvProbe
# 语言：CXX（C++）
project(EnvProbe LANGUAGES CXX)

# 设置C++标准为C++17
# 这确保了编译器支持C++17的所有特性
set(CMAKE_CXX_STANDARD 17)

# 在 MSVC 下强制以 UTF-8 解析与输出源码，避免中文注释导致的解析异常
if(MSVC)
    add_compile_options(/utf-8)
endif()

# ===========================================
# 依赖库查找
# ===========================================

# 查找OpenCL库
# REQUIRED关键字表示OpenCL是必需依赖，如果找不到会报错
# find_package会查找OpenCL头文件和库文件
find_package(OpenCL REQUIRED)

# ===========================================
# 可执行文件目标1：MNIST推理程序（批量 + 分块并行）
# ===========================================

# 创建名为mnist_infer的可执行文件
# 源文件：src/mnist_infer.cpp
# 该程序：单样本前向推理演示（使用 src/mlp_forward.cl 的单核函数）
add_executable(mnist_infer
    src/mnist_infer.cpp
)

# 链接OpenCL库到mnist_infer目标
# mnist_infer程序需要使用OpenCL API进行GPU/CPU计算
target_link_libraries(mnist_infer PRIVATE OpenCL::OpenCL)

# 设置可执行文件的输出目录
# 与ocl_info相同，都输出到构建目录的根目录
set_target_properties(mnist_infer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# 可执行文件目标2：批量/分层核函数主程序 main.cpp
add_executable(mnist_batch
    src/main.cpp
    src/opencl_utils.hpp
)
target_link_libraries(mnist_batch PRIVATE OpenCL::OpenCL)
set_target_properties(mnist_batch PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# 可执行文件目标3：Web Server（一次初始化，多次推理；服务/infer与前端页面）
add_executable(mnist_web
    src/web_server.cpp
    src/infer_engine.hpp
    src/opencl_utils.hpp
)
target_link_libraries(mnist_web PRIVATE OpenCL::OpenCL)
set_target_properties(mnist_web PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# 复制前端静态页面到可执行输出目录
# web 已移动到 src/web/index.html
add_custom_command(TARGET mnist_web POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:mnist_web>/web"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/src/web/index.html" "$<TARGET_FILE_DIR:mnist_web>/web/index.html"
)

# ===========================================
# 构建说明：
# ===========================================
# 1. 该项目包含三个独立的可执行文件：
#    - mnist_infer: 进行MNIST数字识别推理
#    - mnist_batch: 批量MNIST数字识别主程序
#    - mnist_web:   启动本地HTTP服务，前端画板 + 后端双路前向
#
# 2. 两个程序都需要OpenCL运行时支持
#
# 3. 构建方法：
#    - 使用CMake GUI或命令行
#    - 生成的项目文件会自动包含OpenCL依赖
#
# 4. 运行要求：
#    - 安装OpenCL驱动程序
#    - 存在训练好的权重文件（src/model_out/）
#
# ===========================================