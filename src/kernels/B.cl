// ===========================================
// 通用全连接 + 可选 ReLU 的 OpenCL 内核（支持 2D 批量）
// ===========================================
// 形参与布局约定：
// - X  [N, in_dim]  行主序:  X[b*in_dim + i]
// - W  [in_dim, out_dim] 行主序: W[i*out_dim + o]
// - B  [out_dim]
// - Y  [N, out_dim] 行主序:  Y[b*out_dim + o]
// NDRange:
// - global = (gx, gy) 其中 gx >= out_dim 且按 local_x 向上取整；gy = N
// - local  = (lx, 1)  建议 lx=16

#define TILE_SIZE 256  // 可根据设备调整；若本地内存不足，可减小或改为非tiling路径

__kernel void fc_relu_layer(
    __global const float* restrict X,   // [N, in_dim]
    __global const float* restrict W,   // [in_dim, out_dim]
    __global const float* restrict B,   // [out_dim]
    __global float* restrict Y,         // [N, out_dim]
    const int in_dim,
    const int out_dim,
    const int N,
    const int apply_relu                // 1: ReLU, 0: pass-through
)
{
    const int o = get_global_id(0);  // 输出维度索引
    const int b = get_global_id(1);  // batch 索引
    const int valid_o = (o < out_dim);
    const int valid_b = (b < N);

    const __global float* x_row = valid_b ? (X + (size_t)b * in_dim) : X; // b无效时提供安全指针
    float acc = 0.0f;

    // Tiling 版本：把 X 的一段搬到本地内存，提升数据复用（每个 work-group 协作）
    __local float x_tile[TILE_SIZE];
    const int lx = get_local_id(0);
    const int lsize = get_local_size(0);

    for (int t = 0; t < in_dim; t += TILE_SIZE) {
        const int tile_len = (t + TILE_SIZE <= in_dim) ? TILE_SIZE : (in_dim - t);

        // 协作加载 X[b, t:t+tile_len)
        for (int i = lx; i < tile_len; i += lsize) {
            x_tile[i] = valid_b ? x_row[t + i] : 0.0f; // 无效b填0，保证tile内容确定
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 累加: acc += sum_{i in tile} x_tile[i] * W[(t+i), o]（仅有效o做累加）
        if (valid_o) {
            const __global float* w_col_base = W + (size_t)t * out_dim + o; // 指向 W[t, o]
            for (int i = 0; i < tile_len; ++i) {
                acc += x_tile[i] * w_col_base[(size_t)i * out_dim];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 加偏置（仅有效o）
    if (valid_o) {
        acc += B[o];
    }

    // 可选 ReLU
    if (apply_relu && valid_o) {
        acc = fmax(acc, 0.0f);
    }

    // 写回输出
    if (valid_o && valid_b) {
        Y[(size_t)b * out_dim + o] = acc;
    }
}



