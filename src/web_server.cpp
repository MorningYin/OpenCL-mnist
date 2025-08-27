// =============================================================
// 内置 HTTP 服务（Windows Winsock）
// - 提供两种前向方法的统一推理接口 /infer
// - GET / 返回前端页面，用于画板手写与结果展示
// - 启动时一次性加载权重并初始化 OpenCL 与两种前向引擎
// =============================================================

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>
#include <cctype>

#include "infer_engine.hpp"

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
#endif

// 读取文本文件（若不存在返回空字符串）
static std::string read_file_text(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return {};
    std::ostringstream oss; oss << ifs.rdbuf();
    return oss.str();
}

// 解析最简 HTTP 请求（仅支持一段 headers + 可选 body）
//
// 设计目标与限制：
// - 该解析器用于本地演示用途，遵循“足够用”的最小实现：
//   1) 仅处理形如：
//      <METHOD> <PATH> HTTP/1.1\r\n
//      <Header-Name>: <Header-Value>\r\n
//      ...多行Header...
//      \r\n
//      <可选Body>
//   2) 通过“\r\n\r\n”定位头部结束位置；不支持分块传输（chunked transfer-encoding）
//   3) 仅关心 Content-Length（大小写不敏感），其余头部字段均忽略
//   4) 不解析 query string、fragment，也不校验 HTTP 版本号或方法名合法性
//   5) 不支持折叠头（header folding）与重复同名头的合并策略
// - 安全提示：未做 Content-Length 的大小上限校验，生产环境须限制长度并做异常处理。
//
// 参数：
// - req:    已读入的 HTTP 原始报文字符串（通常包含完整起始行与所有头部，可能带部分或全部 body）
// - method: 输出解析到的 HTTP 方法（如 "GET" / "POST"）
// - path:   输出解析到的请求路径（不含协议/主机/查询串）
// - headers:原样返回起始行+头部的字符串（便于调试/日志）
// - body:   输出头部结束后已读到的 body 片段（若服务器主循环后续继续读，则此处可能只是 body 的前缀）
// - content_length: 输出从头部解析到的 Content-Length 数值（若不存在则为 0）
// 返回：
// - true  解析成功（找到了“\r\n\r\n”且成功提取起始行）
// - false 解析失败（未找到头部结束分隔符）
static bool parse_http_request(const std::string& req, std::string& method, std::string& path, std::string& headers, std::string& body, size_t& content_length) {
    content_length = 0;
    // 1) 定位头部结束：HTTP 规范以 CRLFCRLF 分隔 header 与 body
    auto pos = req.find("\r\n\r\n");
    if (pos == std::string::npos) return false;
    // 2) 切分出“起始行+所有头部”（不含末尾 CRLFCRLF）与“已读 body 片段”
    std::string start_line_and_headers = req.substr(0, pos);
    body = req.substr(pos + 4); // 跳过 CRLFCRLF 四个字符

    // 3) 读取起始行：格式约定为 "METHOD SP PATH SP HTTP/VERSION"
    //    这里宽松处理：仅读取前两个 token 作为 method 与 path，忽略版本字段
    std::istringstream iss(start_line_and_headers);
    std::string start_line;
    std::getline(iss, start_line);
    if (!start_line.empty() && start_line.back() == '\r') start_line.pop_back();
    {
        std::istringstream ls(start_line);
        ls >> method >> path; // 忽略 HTTP 版本号；未校验合法性
    }
    headers = start_line_and_headers; // 原样保留，便于调试或日志打印

    // parse content-length
    // 4) 自第二行起逐行读取 header：形如 "Key: Value"，我们做：
    //    - 大小写不敏感匹配 "content-length"
    //    - 去除值两侧空白
    //    - 成功解析为正整数则写入 content_length
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto p = line.find(":");
        if (p != std::string::npos) {
            std::string key = line.substr(0, p);
            std::string val = line.substr(p + 1);
            // trim
            auto ltrim = [](std::string& s){ size_t i=0; while (i<s.size() && (s[i]==' '||s[i]=='\t')) ++i; s.erase(0,i); };
            auto rtrim = [](std::string& s){ while (!s.empty() && (s.back()==' '||s.back()=='\t')) s.pop_back(); };
            ltrim(val); rtrim(val);
            for (auto& c : key) c = (char)tolower(c);
            if (key == "content-length") {
                content_length = (size_t)std::strtoul(val.c_str(), nullptr, 10);
            }
        }
    }
    return true;
}

// 朴素 JSON 解析：提取 key "pixels" 的数组，长度需为 784
static bool parse_pixels_from_json(const std::string& json, std::vector<float>& out784) {
    // naive parser: find first '[' and ']' after "pixels"
    size_t key = json.find("\"pixels\"");
    if (key == std::string::npos) key = json.find("pixels");
    size_t lb = json.find('[', key == std::string::npos ? 0 : key);
    size_t rb = json.find(']', lb == std::string::npos ? 0 : lb);
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return false;
    std::string arr = json.substr(lb + 1, rb - lb - 1);
    std::istringstream iss(arr);
    out784.clear(); out784.reserve(784);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // trim
        size_t s = 0; while (s < token.size() && (token[s]==' '||token[s]=='\t' || token[s]=='\n' || token[s]=='\r')) ++s;
        size_t e = token.size(); while (e> s && (token[e-1]==' '||token[e-1]=='\t'||token[e-1]=='\n'||token[e-1]=='\r')) --e;
        if (e > s) {
            float v = std::strtof(token.substr(s, e-s).c_str(), nullptr);
            out784.push_back(v);
        }
    }
    return out784.size() == 784;
}

// JSON 字符串转义（本服务仅少量使用）
static std::string json_escape(const std::string& s) {
    std::string o; o.reserve(s.size()+8);
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c; break;
        }
    }
    return o;
}

// 发送 HTTP 响应（附带 CORS 头，便于前端调试）
static void send_http_response(SOCKET client, const std::string& status, const std::string& content_type, const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << "\r\n";
    oss << "Content-Type: " << content_type << "\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n"; // helpful during dev
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Connection: close\r\n\r\n";
    std::string head = oss.str();
    send(client, head.c_str(), (int)head.size(), 0);
    if (!body.empty()) {
        send(client, body.c_str(), (int)body.size(), 0);
    }
}

int main() {
    try {
        // Load weights once
        ModelWeights weights;
        std::vector<std::string> roots = {"src/model_out/", "../src/model_out/", "../../src/model_out/"};
        std::string root;
        for (const auto& r : roots) {
            std::ifstream test(r + "linear0_W.txt");
            if (test.is_open()) { root = r; break; }
        }
        if (root.empty()) throw std::runtime_error("Cannot locate model weights under src/model_out");
        read_matrix_txt(root + "linear0_W.txt", 784, 128, weights.W0);
        read_vector_txt(root + "linear0_b.txt", 128, weights.b0);
        read_matrix_txt(root + "linear1_W.txt", 128, 64, weights.W1);
        read_vector_txt(root + "linear1_b.txt", 64,  weights.b1);
        read_matrix_txt(root + "linear2_W.txt", 64,  10, weights.W2);
        read_vector_txt(root + "linear2_b.txt", 10,  weights.b2);

        // Initialize OpenCL once and both forward methods
        OpenCLContext clctx; clctx.init();
        ForwardSingleKernel methodA; methodA.initialize(clctx, weights);
        ForwardLayerKernels methodB; methodB.initialize(clctx, weights);

        // Prepare Winsock
#ifdef _WIN32
        WSADATA wsaData; if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) { throw std::runtime_error("WSAStartup failed"); }
        SOCKET server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (server == INVALID_SOCKET) throw std::runtime_error("socket failed");
        u_long nonblock = 0; // keep blocking
        ioctlsocket(server, FIONBIO, &nonblock);

        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(8080); addr.sin_addr.s_addr = htonl(INADDR_ANY);
        int opt = 1; setsockopt(server, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
        if (bind(server, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) throw std::runtime_error("bind failed");
        if (listen(server, SOMAXCONN) == SOCKET_ERROR) throw std::runtime_error("listen failed");
        std::cerr << "HTTP server listening on http://127.0.0.1:8080\n";

        // ============================= 主循环 =============================
        // 单线程、同步处理：每次处理一个到来的 TCP 连接，请求-响应后关闭
        // 说明：当前套接字是阻塞模式（nonblock=0），accept 将阻塞直到有连接到来。
        // ==================================================================
        while (true) {
            sockaddr_in caddr{}; int clen = sizeof(caddr);
            SOCKET client = accept(server, (sockaddr*)&caddr, &clen);
            if (client == INVALID_SOCKET) {
                int e = WSAGetLastError();
                // 注：阻塞模式下通常不会出现 WSAEWOULDBLOCK，此处仅作为容错处理
                if (e == WSAEWOULDBLOCK) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; }
                std::cerr << "accept error: " << e << std::endl; continue;
            }

            std::string req; req.reserve(4096);
            char buf[4096];
            int n = 0;
            // 读取请求头：HTTP 报文以 "\r\n\r\n" 分隔头与体
            // 我们持续 recv 直到遇到空行分隔符，或连接关闭
            while ((n = recv(client, buf, (int)sizeof(buf), 0)) > 0) {
                req.append(buf, buf + n);
                if (req.find("\r\n\r\n") != std::string::npos) break;
            }
            if (n <= 0) { closesocket(client); continue; }

            std::string method, path, headers, body; size_t content_length = 0;
            // 粗略解析请求行与头，并从 Content-Length 获取应读入的请求体长度
            if (!parse_http_request(req, method, path, headers, body, content_length)) {
                send_http_response(client, "400 Bad Request", "text/plain; charset=utf-8", "Bad Request");
                closesocket(client); continue;
            }

            // 若已读入的 body 不足 Content-Length，继续读取剩余字节
            // 注意：此处未处理分块传输编码，仅支持带 Content-Length 的简单 POST
            size_t have = body.size();
            while (have < content_length) {
                size_t rem = content_length - have;
                size_t toReadSz = (sizeof(buf) < rem) ? sizeof(buf) : rem;
                int toRead = (int)toReadSz;
                n = recv(client, buf, toRead, 0);
                if (n <= 0) break; body.append(buf, buf+n); have += (size_t)n;
            }

            if (method == "GET" && (path == "/" || path == "/index.html")) {
                // 静态页面：前端画板，用于采集手写并发起 /infer 调用
                std::string html = read_file_text("src/web/index.html");
                if (html.empty()) html = "<html><body><h1>Index not found</h1></body></html>";
                send_http_response(client, "200 OK", "text/html; charset=utf-8", html);
            } else if (method == "POST" && path == "/infer") {
                // 推理接口：body JSON 形如 {"pixels":[...784 floats in [0,1]...]}
                std::vector<float> x784;
                if (!parse_pixels_from_json(body, x784)) {
                    send_http_response(client, "400 Bad Request", "application/json", "{\"error\":\"invalid body\"}");
                    closesocket(client); continue;
                }
                // 安全归一化/裁剪：确保像素值在 [0,1]
                for (float& v : x784) { if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f; }

                // 调用两条推理路径，并分别计时（不包含 JSON 解析/序列化耗时）
                std::array<float,10> la{}; std::array<float,10> lb{};
                auto t0 = std::chrono::high_resolution_clock::now();
                methodA.infer(x784, la);
                auto t1 = std::chrono::high_resolution_clock::now();
                methodB.infer(x784, lb);
                auto t2 = std::chrono::high_resolution_clock::now();
                double msA = std::chrono::duration<double, std::milli>(t1 - t0).count();
                double msB = std::chrono::duration<double, std::milli>(t2 - t1).count();
                int predA = argmax10(la.data());
                int predB = argmax10(lb.data());

                // 统一响应：返回两种方法的预测与耗时（毫秒）
                std::ostringstream json;
                json << "{\"methodA\":{\"pred\":" << predA << ",\"ms\":" << msA << "},";
                json << "\"methodB\":{\"pred\":" << predB << ",\"ms\":" << msB << "}}";
                send_http_response(client, "200 OK", "application/json", json.str());
            } else if (method == "OPTIONS") {
                // 预检请求（CORS）：返回允许的方法与请求头
                std::ostringstream oss; oss << "HTTP/1.1 204 No Content\r\n";
                oss << "Access-Control-Allow-Origin: *\r\n";
                oss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
                oss << "Access-Control-Allow-Headers: Content-Type\r\n";
                oss << "Content-Length: 0\r\n\r\n";
                std::string h = oss.str(); send(client, h.c_str(), (int)h.size(), 0);
            } else {
                // 未匹配的路径或方法
                send_http_response(client, "404 Not Found", "text/plain; charset=utf-8", "Not Found");
            }

            // 短连接：每个请求结束后立即关闭
            closesocket(client);
        }
        closesocket(server);
        WSACleanup();
#else
        std::cerr << "This demo server currently supports Windows only." << std::endl;
        return 1;
#endif
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}


