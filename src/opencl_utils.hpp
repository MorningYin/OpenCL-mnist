#pragma once

// =============================================================
// OpenCL 常用工具函数
// - 统一的错误检查与异常抛出
// - 简单的设备选择与打印（选择第一个可用设备）
// - 获取 OpenCL 程序构建日志
// =============================================================

#include <CL/cl.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

// 错误检查：若 err != CL_SUCCESS，则输出消息并抛出异常
inline void checkCLError(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << msg << ", OpenCL error = " << err << std::endl;
        throw std::runtime_error(std::string(msg) + ", OpenCL error = " + std::to_string(err));
    }
}

// 选择并打印第一个可用的 OpenCL 设备，同时输出平台与设备名
inline cl_device_id pick_first_device_print() {
    cl_uint numPlatforms = 0;
    checkCLError(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs count");
    if (numPlatforms == 0) {
        throw std::runtime_error("No OpenCL platforms");
    }
    std::vector<cl_platform_id> plats(numPlatforms);
    checkCLError(clGetPlatformIDs(numPlatforms, plats.data(), nullptr), "clGetPlatformIDs list");

    for (cl_uint p = 0; p < numPlatforms; ++p) {
        size_t sz = 0; clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, 0, nullptr, &sz);
        std::string pname(sz, '\0'); clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sz, pname.data(), nullptr);
        std::cerr << "Platform[" << p << "]: " << pname.c_str() << std::endl;

        cl_uint nd = 0; clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
        for (cl_uint i = 0; i < nd; ++i) {
            sz = 0; clGetDeviceInfo(devs[i], CL_DEVICE_NAME, 0, nullptr, &sz);
            std::string dname(sz, '\0'); clGetDeviceInfo(devs[i], CL_DEVICE_NAME, sz, dname.data(), nullptr);
            std::cerr << "  Device[" << i << "]: " << dname.c_str() << std::endl;
            return devs[i];
        }
    }
    throw std::runtime_error("No OpenCL devices");
}

// 获取指定程序与设备的构建日志（用于调试内核编译失败等问题）
inline std::string build_log(cl_program prog, cl_device_id dev) {
    size_t logSize = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    std::string log(logSize, '\0');
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
    return log;
}


