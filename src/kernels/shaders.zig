pub const common =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\constant uint SIMD_SIZE = 32;
    \\
    \\inline float fast_silu(float x) {
    \\    return x / (1.0f + exp(-x));
    \\}
    \\
    \\inline float fast_rsqrt(float x) {
    \\    return rsqrt(x);
    \\}
;

pub const rmsnorm = common ++
    \\
    \\kernel void rmsnorm(
    \\    device const float* x [[buffer(0)]],
    \\    device const float* weight [[buffer(1)]],
    \\    device float* out [[buffer(2)]],
    \\    constant uint& dim [[buffer(3)]],
    \\    constant float& eps [[buffer(4)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]],
    \\    ushort simd_gid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    threadgroup float shared[8];
    \\
    \\    uint offset = tgid * dim;
    \\    float local_sum = 0.0f;
    \\
    \\    for (uint i = tid; i < dim; i += 256) {
    \\        float val = x[offset + i];
    \\        local_sum += val * val;
    \\    }
    \\
    \\    float simd_sum_val = simd_sum(local_sum);
    \\    if (simd_idx == 0) shared[simd_gid] = simd_sum_val;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    if (simd_gid == 0) {
    \\        float total = (simd_idx < 8) ? shared[simd_idx] : 0.0f;
    \\        total = simd_sum(total);
    \\        if (simd_idx == 0) shared[0] = total;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float rms = fast_rsqrt(shared[0] / float(dim) + eps);
    \\
    \\    for (uint i = tid; i < dim; i += 256) {
    \\        out[offset + i] = x[offset + i] * rms * weight[i];
    \\    }
    \\}
;

pub const rope = common ++
    \\
    \\kernel void rope(
    \\    device const float* x [[buffer(0)]],
    \\    device const float* cos_cache [[buffer(1)]],
    \\    device const float* sin_cache [[buffer(2)]],
    \\    device float* out [[buffer(3)]],
    \\    constant uint& dim [[buffer(4)]],
    \\    constant uint& head_dim [[buffer(5)]],
    \\    constant uint& pos [[buffer(6)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]]
    \\) {
    \\    uint half_head = head_dim / 2;
    \\    uint offset = tgid * dim;
    \\
    \\    for (uint i = tid; i < dim; i += 256) {
    \\        uint pos_in_head = i % head_dim;
    \\        float x_val = x[offset + i];
    \\
    \\        if (pos_in_head < half_head) {
    \\            float x_pair = x[offset + i + half_head];
    \\            float cos_v = cos_cache[pos * half_head + pos_in_head];
    \\            float sin_v = sin_cache[pos * half_head + pos_in_head];
    \\            out[offset + i] = x_val * cos_v - x_pair * sin_v;
    \\        } else {
    \\            uint freq_idx = pos_in_head - half_head;
    \\            float x_pair = x[offset + i - half_head];
    \\            float cos_v = cos_cache[pos * half_head + freq_idx];
    \\            float sin_v = sin_cache[pos * half_head + freq_idx];
    \\            out[offset + i] = x_val * cos_v + x_pair * sin_v;
    \\        }
    \\    }
    \\}
;

pub const swiglu = common ++
    \\
    \\kernel void swiglu(
    \\    device const float* x [[buffer(0)]],
    \\    device const float* gate_weight [[buffer(1)]],
    \\    device const float* up_weight [[buffer(2)]],
    \\    device float* out [[buffer(3)]],
    \\    constant uint& in_dim [[buffer(4)]],
    \\    constant uint& out_dim [[buffer(5)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]]
    \\) {
    \\    uint x_offset = tgid * in_dim;
    \\    uint out_offset = tgid * out_dim;
    \\
    \\    for (uint o = tid; o < out_dim; o += 256) {
    \\        float gate_acc = 0.0f;
    \\        float up_acc = 0.0f;
    \\
    \\        for (uint i = 0; i < in_dim; i++) {
    \\            float xv = x[x_offset + i];
    \\            gate_acc += xv * gate_weight[i * out_dim + o];
    \\            up_acc += xv * up_weight[i * out_dim + o];
    \\        }
    \\
    \\        out[out_offset + o] = fast_silu(gate_acc) * up_acc;
    \\    }
    \\}
;

pub const fused_rmsnorm_rope = common ++
    \\
    \\kernel void fused_rmsnorm_rope(
    \\    device const float* x [[buffer(0)]],
    \\    device const float* weight [[buffer(1)]],
    \\    device const float* cos_cache [[buffer(2)]],
    \\    device const float* sin_cache [[buffer(3)]],
    \\    device float* out [[buffer(4)]],
    \\    constant uint& dim [[buffer(5)]],
    \\    constant uint& head_dim [[buffer(6)]],
    \\    constant uint& pos [[buffer(7)]],
    \\    constant float& eps [[buffer(8)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]],
    \\    ushort simd_gid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    threadgroup float shared[8];
    \\    uint offset = tgid * dim;
    \\    uint half_head = head_dim / 2;
    \\
    \\    float local_sum = 0.0f;
    \\    for (uint i = tid; i < dim; i += 256) {
    \\        float val = x[offset + i];
    \\        local_sum += val * val;
    \\    }
    \\
    \\    float simd_sum_val = simd_sum(local_sum);
    \\    if (simd_idx == 0) shared[simd_gid] = simd_sum_val;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    if (simd_gid == 0) {
    \\        float total = (simd_idx < 8) ? shared[simd_idx] : 0.0f;
    \\        total = simd_sum(total);
    \\        if (simd_idx == 0) shared[0] = total;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float rms = fast_rsqrt(shared[0] / float(dim) + eps);
    \\
    \\    for (uint i = tid; i < dim; i += 256) {
    \\        float normed = x[offset + i] * rms * weight[i];
    \\        uint pos_in_head = i % head_dim;
    \\
    \\        if (pos_in_head < half_head) {
    \\            float pair_val = x[offset + i + half_head] * rms * weight[i + half_head];
    \\            float cos_v = cos_cache[pos * half_head + pos_in_head];
    \\            float sin_v = sin_cache[pos * half_head + pos_in_head];
    \\            out[offset + i] = normed * cos_v - pair_val * sin_v;
    \\        } else {
    \\            float pair_val = x[offset + i - half_head] * rms * weight[i - half_head];
    \\            uint freq_idx = pos_in_head - half_head;
    \\            float cos_v = cos_cache[pos * half_head + freq_idx];
    \\            float sin_v = sin_cache[pos * half_head + freq_idx];
    \\            out[offset + i] = normed * cos_v + pair_val * sin_v;
    \\        }
    \\    }
    \\}
;

pub const matvec = common ++
    \\
    \\kernel void matvec(
    \\    device const float* x [[buffer(0)]],
    \\    device const float* W [[buffer(1)]],
    \\    device float* out [[buffer(2)]],
    \\    constant uint& K [[buffer(3)]],
    \\    constant uint& N [[buffer(4)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]],
    \\    ushort simd_gid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    threadgroup float shared_x[4096];
    \\
    \\    for (uint i = tid; i < K; i += 256) {
    \\        shared_x[i] = x[i];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    const uint OUTPUTS_PER_THREAD = 4;
    \\    uint n_base = (tgid * 256 + tid) * OUTPUTS_PER_THREAD;
    \\
    \\    float acc[OUTPUTS_PER_THREAD] = {0, 0, 0, 0};
    \\
    \\    for (uint k = 0; k < K; k++) {
    \\        float xv = shared_x[k];
    \\        uint w_row = k * N;
    \\        for (uint i = 0; i < OUTPUTS_PER_THREAD; i++) {
    \\            uint n = n_base + i;
    \\            if (n < N) {
    \\                acc[i] += xv * W[w_row + n];
    \\            }
    \\        }
    \\    }
    \\
    \\    for (uint i = 0; i < OUTPUTS_PER_THREAD; i++) {
    \\        uint n = n_base + i;
    \\        if (n < N) {
    \\            out[n] = acc[i];
    \\        }
    \\    }
    \\}
;

pub const matvec_dual = common ++
    \\
    \\kernel void matvec_dual(
    \\    device const float* x [[buffer(0)]],
    \\    device const float* W1 [[buffer(1)]],
    \\    device const float* W2 [[buffer(2)]],
    \\    device float* out1 [[buffer(3)]],
    \\    device float* out2 [[buffer(4)]],
    \\    constant uint& K [[buffer(5)]],
    \\    constant uint& N [[buffer(6)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]],
    \\    ushort simd_gid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    uint n = tgid * 8 + simd_gid;
    \\    if (n >= N) return;
    \\
    \\    float acc1 = 0.0f;
    \\    float acc2 = 0.0f;
    \\    uint w_offset = n;
    \\
    \\    for (uint k = simd_idx; k < K; k += 32) {
    \\        float xv = x[k];
    \\        acc1 += xv * W1[k * N + w_offset];
    \\        acc2 += xv * W2[k * N + w_offset];
    \\    }
    \\
    \\    acc1 = simd_sum(acc1);
    \\    acc2 = simd_sum(acc2);
    \\    if (simd_idx == 0) {
    \\        out1[n] = acc1;
    \\        out2[n] = acc2;
    \\    }
    \\}
;

pub const simdgroup_matmul = common ++
    \\
    \\using namespace metal;
    \\
    \\kernel void simdgroup_matmul(
    \\    device const float* A [[buffer(0)]],
    \\    device const float* B [[buffer(1)]],
    \\    device float* C [[buffer(2)]],
    \\    constant uint& M [[buffer(3)]],
    \\    constant uint& N [[buffer(4)]],
    \\    constant uint& K [[buffer(5)]],
    \\    uint2 tid [[thread_position_in_threadgroup]],
    \\    uint2 tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]],
    \\    ushort simd_gid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint TILE_M = 32;
    \\    const uint TILE_N = 32;
    \\    const uint TILE_K = 32;
    \\
    \\    threadgroup float As[TILE_M][TILE_K];
    \\    threadgroup float Bs[TILE_K][TILE_N];
    \\
    \\    uint row = tgid.y * TILE_M;
    \\    uint col = tgid.x * TILE_N;
    \\
    \\    float acc[4][4] = {{0}};
    \\
    \\    uint local_row = tid.y;
    \\    uint local_col = tid.x;
    \\
    \\    for (uint k_tile = 0; k_tile < K; k_tile += TILE_K) {
    \\        for (uint i = local_row; i < TILE_M; i += 8) {
    \\            for (uint j = local_col; j < TILE_K; j += 32) {
    \\                uint a_row = row + i;
    \\                uint a_col = k_tile + j;
    \\                As[i][j] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
    \\            }
    \\        }
    \\        for (uint i = local_row; i < TILE_K; i += 8) {
    \\            for (uint j = local_col; j < TILE_N; j += 32) {
    \\                uint b_row = k_tile + i;
    \\                uint b_col = col + j;
    \\                Bs[i][j] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        for (uint k = 0; k < TILE_K; k++) {
    \\            float a_vals[4];
    \\            float b_vals[4];
    \\            for (uint i = 0; i < 4; i++) {
    \\                a_vals[i] = As[local_row * 4 + i][k];
    \\                b_vals[i] = Bs[k][local_col * 4 + i];
    \\            }
    \\            for (uint i = 0; i < 4; i++) {
    \\                for (uint j = 0; j < 4; j++) {
    \\                    acc[i][j] += a_vals[i] * b_vals[j];
    \\                }
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    for (uint i = 0; i < 4; i++) {
    \\        for (uint j = 0; j < 4; j++) {
    \\            uint out_row = row + local_row * 4 + i;
    \\            uint out_col = col + local_col * 4 + j;
    \\            if (out_row < M && out_col < N) {
    \\                C[out_row * N + out_col] = acc[i][j];
    \\            }
    \\        }
    \\    }
    \\}
;

pub const fused_silu_mul = common ++
    \\
    \\kernel void fused_silu_mul(
    \\    device const float* gate [[buffer(0)]],
    \\    device const float* up [[buffer(1)]],
    \\    device float* out [[buffer(2)]],
    \\    constant uint& n [[buffer(3)]],
    \\    uint tid [[thread_position_in_grid]]
    \\) {
    \\    if (tid >= n) return;
    \\    float g = gate[tid];
    \\    out[tid] = fast_silu(g) * up[tid];
    \\}
;

pub const fused_rmsnorm_rope_swiglu = common ++
    \\
    \\kernel void fused_rmsnorm_rope_swiglu(
    \\    device const float* x [[buffer(0)]],
    \\    device const float* norm_weight [[buffer(1)]],
    \\    device const float* cos_cache [[buffer(2)]],
    \\    device const float* sin_cache [[buffer(3)]],
    \\    device const float* gate_weight [[buffer(4)]],
    \\    device const float* up_weight [[buffer(5)]],
    \\    device float* out [[buffer(6)]],
    \\    constant uint& hidden_dim [[buffer(7)]],
    \\    constant uint& inter_dim [[buffer(8)]],
    \\    constant uint& head_dim [[buffer(9)]],
    \\    constant uint& pos [[buffer(10)]],
    \\    constant float& eps [[buffer(11)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]],
    \\    ushort simd_gid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    threadgroup float shared_reduce[8];
    \\    threadgroup float shared_normed[4096];
    \\
    \\    uint x_offset = tgid * hidden_dim;
    \\    uint out_offset = tgid * inter_dim;
    \\    uint half_head = head_dim / 2;
    \\
    \\    float local_sum = 0.0f;
    \\    for (uint i = tid; i < hidden_dim; i += 256) {
    \\        float val = x[x_offset + i];
    \\        local_sum += val * val;
    \\    }
    \\
    \\    float simd_sum_val = simd_sum(local_sum);
    \\    if (simd_idx == 0) shared_reduce[simd_gid] = simd_sum_val;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    if (simd_gid == 0) {
    \\        float total = (simd_idx < 8) ? shared_reduce[simd_idx] : 0.0f;
    \\        total = simd_sum(total);
    \\        if (simd_idx == 0) shared_reduce[0] = total;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float rms = fast_rsqrt(shared_reduce[0] / float(hidden_dim) + eps);
    \\
    \\    for (uint i = tid; i < hidden_dim; i += 256) {
    \\        float x_val = x[x_offset + i];
    \\        float normed = x_val * rms * norm_weight[i];
    \\
    \\        uint pos_in_head = i % head_dim;
    \\        if (pos_in_head < half_head) {
    \\            uint pair_idx = i + half_head;
    \\            float x_pair = x[x_offset + pair_idx] * rms * norm_weight[pair_idx];
    \\            float cos_v = cos_cache[pos * half_head + pos_in_head];
    \\            float sin_v = sin_cache[pos * half_head + pos_in_head];
    \\            shared_normed[i] = normed * cos_v - x_pair * sin_v;
    \\        } else {
    \\            uint pair_idx = i - half_head;
    \\            float x_pair = x[x_offset + pair_idx] * rms * norm_weight[pair_idx];
    \\            uint freq_idx = pos_in_head - half_head;
    \\            float cos_v = cos_cache[pos * half_head + freq_idx];
    \\            float sin_v = sin_cache[pos * half_head + freq_idx];
    \\            shared_normed[i] = normed * cos_v + x_pair * sin_v;
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (uint o = tid; o < inter_dim; o += 256) {
    \\        float gate_acc = 0.0f;
    \\        float up_acc = 0.0f;
    \\
    \\        for (uint h = 0; h < hidden_dim; h++) {
    \\            float xn = shared_normed[h];
    \\            gate_acc += xn * gate_weight[h * inter_dim + o];
    \\            up_acc += xn * up_weight[h * inter_dim + o];
    \\        }
    \\
    \\        out[out_offset + o] = fast_silu(gate_acc) * up_acc;
    \\    }
    \\}
;

pub const qmatmul_int4 = common ++
    \\
    \\kernel void qmatmul_int4(
    \\    device const float* x [[buffer(0)]],
    \\    device const uchar* w_packed [[buffer(1)]],
    \\    device const half* scales [[buffer(2)]],
    \\    device const uchar* zeros [[buffer(3)]],
    \\    device float* out [[buffer(4)]],
    \\    constant uint& M [[buffer(5)]],
    \\    constant uint& K [[buffer(6)]],
    \\    constant uint& N [[buffer(7)]],
    \\    constant uint& group_size [[buffer(8)]],
    \\    uint2 tid [[thread_position_in_threadgroup]],
    \\    uint2 tgid [[threadgroup_position_in_grid]]
    \\) {
    \\    uint m = tgid.y;
    \\    uint n_base = tgid.x * 16;
    \\    uint n = n_base + tid.x;
    \\
    \\    if (m >= M || n >= N) return;
    \\
    \\    float acc = 0.0f;
    \\    uint x_offset = m * K;
    \\
    \\    for (uint k = 0; k < K; k++) {
    \\        uint byte_idx = (k / 2) * N + n;
    \\        uchar packed = w_packed[byte_idx];
    \\        int w_int = (k % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    \\
    \\        uint group = k / group_size;
    \\        float scale = float(scales[group * N + n]);
    \\
    \\        uint zero_byte_idx = group * (N / 2) + n / 2;
    \\        uchar zero_packed = zeros[zero_byte_idx];
    \\        int zero = (n % 2 == 0) ? (zero_packed & 0x0F) : ((zero_packed >> 4) & 0x0F);
    \\
    \\        float w_float = scale * float(w_int - zero);
    \\        acc += x[x_offset + k] * w_float;
    \\    }
    \\
    \\    out[m * N + n] = acc;
    \\}
;

pub const paged_attention = common ++
    \\
    \\constant uint KV_BLOCK_SIZE = 16;
    \\
    \\kernel void paged_attention(
    \\    device const float* Q [[buffer(0)]],
    \\    device const float* k_cache [[buffer(1)]],
    \\    device const float* v_cache [[buffer(2)]],
    \\    device const int* block_tables [[buffer(3)]],
    \\    device const int* context_lens [[buffer(4)]],
    \\    device float* O [[buffer(5)]],
    \\    constant uint& num_heads [[buffer(6)]],
    \\    constant uint& head_dim [[buffer(7)]],
    \\    constant uint& max_blocks [[buffer(8)]],
    \\    constant float& scale [[buffer(9)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint2 tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]]
    \\) {
    \\    uint batch_idx = tgid.y;
    \\    uint head_idx = tgid.x;
    \\    uint context_len = context_lens[batch_idx];
    \\    uint num_blocks = (context_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    \\
    \\    threadgroup float shared_q[128];
    \\    threadgroup float shared_k[KV_BLOCK_SIZE][128];
    \\    threadgroup float shared_v[KV_BLOCK_SIZE][128];
    \\
    \\    uint q_offset = (batch_idx * num_heads + head_idx) * head_dim;
    \\    for (uint d = tid; d < head_dim; d += 256) {
    \\        shared_q[d] = Q[q_offset + d];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float max_score = -INFINITY;
    \\    float sum_exp = 0.0f;
    \\    float acc[4] = {0, 0, 0, 0};
    \\
    \\    for (uint block_idx = 0; block_idx < num_blocks; block_idx++) {
    \\        int phys_block = block_tables[batch_idx * max_blocks + block_idx];
    \\        uint block_offset = (phys_block * num_heads + head_idx) * KV_BLOCK_SIZE * head_dim;
    \\
    \\        for (uint i = tid; i < KV_BLOCK_SIZE * head_dim; i += 256) {
    \\            uint kv_idx = i / head_dim;
    \\            uint d = i % head_dim;
    \\            shared_k[kv_idx][d] = k_cache[block_offset + i];
    \\            shared_v[kv_idx][d] = v_cache[block_offset + i];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        uint block_start = block_idx * KV_BLOCK_SIZE;
    \\        for (uint kv_idx = 0; kv_idx < KV_BLOCK_SIZE; kv_idx++) {
    \\            uint global_kv = block_start + kv_idx;
    \\            if (global_kv >= context_len) break;
    \\
    \\            float score = 0.0f;
    \\            for (uint d = simd_idx; d < head_dim; d += 32) {
    \\                score += shared_q[d] * shared_k[kv_idx][d];
    \\            }
    \\            score = simd_sum(score) * scale;
    \\
    \\            float new_max = max(max_score, score);
    \\            float scale_factor = exp(max_score - new_max);
    \\            float p = exp(score - new_max);
    \\
    \\            for (uint i = 0; i < 4; i++) {
    \\                acc[i] = acc[i] * scale_factor + p * shared_v[kv_idx][simd_idx + i * 32];
    \\            }
    \\
    \\            sum_exp = sum_exp * scale_factor + p;
    \\            max_score = new_max;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    float inv_sum = 1.0f / sum_exp;
    \\    uint out_offset = (batch_idx * num_heads + head_idx) * head_dim;
    \\    for (uint i = 0; i < 4; i++) {
    \\        uint d = simd_idx + i * 32;
    \\        if (d < head_dim) {
    \\            O[out_offset + d] = acc[i] * inv_sum;
    \\        }
    \\    }
    \\}
;

pub const gqa_attention = common ++
    \\
    \\kernel void gqa_attention(
    \\    device const float* Q [[buffer(0)]],
    \\    device const float* K [[buffer(1)]],
    \\    device const float* V [[buffer(2)]],
    \\    device float* O [[buffer(3)]],
    \\    constant uint& batch_size [[buffer(4)]],
    \\    constant uint& num_heads [[buffer(5)]],
    \\    constant uint& num_kv_heads [[buffer(6)]],
    \\    constant uint& seq_len [[buffer(7)]],
    \\    constant uint& head_dim [[buffer(8)]],
    \\    constant float& scale [[buffer(9)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint3 tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]]
    \\) {
    \\    uint batch_idx = tgid.z;
    \\    uint head_idx = tgid.y;
    \\    uint q_block = tgid.x;
    \\
    \\    uint heads_per_group = num_heads / num_kv_heads;
    \\    uint kv_head_idx = head_idx / heads_per_group;
    \\
    \\    uint q_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    \\    uint kv_offset = (batch_idx * num_kv_heads + kv_head_idx) * seq_len * head_dim;
    \\
    \\    constant uint BLOCK_Q = 16;
    \\    constant uint BLOCK_KV = 64;
    \\    uint q_start = q_block * BLOCK_Q;
    \\
    \\    threadgroup float shared_Q[BLOCK_Q][128];
    \\    threadgroup float shared_K[BLOCK_KV][128];
    \\    threadgroup float shared_V[BLOCK_KV][128];
    \\
    \\    for (uint i = tid; i < BLOCK_Q * head_dim; i += 256) {
    \\        uint qi = i / head_dim;
    \\        uint d = i % head_dim;
    \\        uint gq = q_start + qi;
    \\        if (gq < seq_len) shared_Q[qi][d] = Q[q_offset + gq * head_dim + d];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float row_max[BLOCK_Q];
    \\    float row_sum[BLOCK_Q];
    \\    float acc[BLOCK_Q][128];
    \\
    \\    for (uint i = 0; i < BLOCK_Q; i++) {
    \\        row_max[i] = -INFINITY;
    \\        row_sum[i] = 0.0f;
    \\        for (uint d = 0; d < head_dim; d++) acc[i][d] = 0.0f;
    \\    }
    \\
    \\    for (uint kv_block = 0; kv_block * BLOCK_KV < seq_len; kv_block++) {
    \\        uint kv_start = kv_block * BLOCK_KV;
    \\
    \\        for (uint i = tid; i < BLOCK_KV * head_dim; i += 256) {
    \\            uint ki = i / head_dim;
    \\            uint d = i % head_dim;
    \\            uint gk = kv_start + ki;
    \\            if (gk < seq_len) {
    \\                shared_K[ki][d] = K[kv_offset + gk * head_dim + d];
    \\                shared_V[ki][d] = V[kv_offset + gk * head_dim + d];
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        for (uint qi = 0; qi < BLOCK_Q; qi++) {
    \\            uint gq = q_start + qi;
    \\            if (gq >= seq_len) continue;
    \\
    \\            float local_max = row_max[qi];
    \\            for (uint ki = 0; ki < BLOCK_KV; ki++) {
    \\                uint gk = kv_start + ki;
    \\                if (gk >= seq_len) break;
    \\
    \\                float score = 0.0f;
    \\                for (uint d = 0; d < head_dim; d++) {
    \\                    score += shared_Q[qi][d] * shared_K[ki][d];
    \\                }
    \\                score *= scale;
    \\                if (gq < gk) score = -INFINITY;
    \\                local_max = max(local_max, score);
    \\            }
    \\
    \\            float sf = exp(row_max[qi] - local_max);
    \\            for (uint d = 0; d < head_dim; d++) acc[qi][d] *= sf;
    \\            row_sum[qi] *= sf;
    \\            row_max[qi] = local_max;
    \\
    \\            for (uint ki = 0; ki < BLOCK_KV; ki++) {
    \\                uint gk = kv_start + ki;
    \\                if (gk >= seq_len) break;
    \\
    \\                float score = 0.0f;
    \\                for (uint d = 0; d < head_dim; d++) {
    \\                    score += shared_Q[qi][d] * shared_K[ki][d];
    \\                }
    \\                score *= scale;
    \\                if (gq < gk) score = -INFINITY;
    \\
    \\                float p = exp(score - local_max);
    \\                row_sum[qi] += p;
    \\                for (uint d = 0; d < head_dim; d++) {
    \\                    acc[qi][d] += p * shared_V[ki][d];
    \\                }
    \\            }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    for (uint qi = 0; qi < BLOCK_Q; qi++) {
    \\        uint gq = q_start + qi;
    \\        if (gq >= seq_len) continue;
    \\        float inv = 1.0f / row_sum[qi];
    \\        for (uint d = tid; d < head_dim; d += 256) {
    \\            O[q_offset + gq * head_dim + d] = acc[qi][d] * inv;
    \\        }
    \\    }
    \\}
;

pub const flash_attention_fwd = common ++
    \\
    \\kernel void flash_attention_fwd(
    \\    device const float* Q [[buffer(0)]],
    \\    device const float* K [[buffer(1)]],
    \\    device const float* V [[buffer(2)]],
    \\    device float* O [[buffer(3)]],
    \\    constant uint& seq_len [[buffer(4)]],
    \\    constant uint& head_dim [[buffer(5)]],
    \\    constant float& scale [[buffer(6)]],
    \\    uint tid [[thread_position_in_threadgroup]],
    \\    uint tgid [[threadgroup_position_in_grid]],
    \\    ushort simd_idx [[thread_index_in_simdgroup]],
    \\    ushort simd_gid [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    uint batch_head = tgid / ((seq_len + 15) / 16);
    \\    uint q_idx = (tgid % ((seq_len + 15) / 16)) * 16 + tid / 16;
    \\    if (q_idx >= seq_len) return;
    \\
    \\    uint qkv_offset = batch_head * seq_len * head_dim;
    \\    uint q_offset = qkv_offset + q_idx * head_dim;
    \\
    \\    threadgroup float shared_q[256];
    \\    threadgroup float shared_scores[16];
    \\
    \\    for (uint d = tid; d < head_dim; d += 256) {
    \\        shared_q[d] = Q[q_offset + d];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    float row_max = -INFINITY;
    \\    float row_sum = 0.0f;
    \\    float acc[4] = {0, 0, 0, 0};
    \\
    \\    for (uint kv_start = 0; kv_start < seq_len; kv_start += 16) {
    \\        uint k_idx = kv_start + simd_idx % 16;
    \\        float score = 0.0f;
    \\
    \\        if (k_idx < seq_len && k_idx <= q_idx) {
    \\            uint k_offset = qkv_offset + k_idx * head_dim;
    \\            for (uint d = 0; d < head_dim; d++) {
    \\                score += shared_q[d] * K[k_offset + d];
    \\            }
    \\            score *= scale;
    \\        } else {
    \\            score = -INFINITY;
    \\        }
    \\
    \\        if (simd_gid == 0 && simd_idx < 16) {
    \\            shared_scores[simd_idx] = score;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\        for (uint ki = 0; ki < 16; ki++) {
    \\            uint global_k = kv_start + ki;
    \\            if (global_k >= seq_len || global_k > q_idx) break;
    \\
    \\            float s = shared_scores[ki];
    \\            float new_max = max(row_max, s);
    \\            float scale_old = exp(row_max - new_max);
    \\            float p = exp(s - new_max);
    \\
    \\            uint v_offset = qkv_offset + global_k * head_dim;
    \\            for (uint i = 0; i < 4; i++) {
    \\                uint d = simd_idx + i * 32;
    \\                if (d < head_dim) {
    \\                    acc[i] = acc[i] * scale_old + p * V[v_offset + d];
    \\                }
    \\            }
    \\            row_sum = row_sum * scale_old + p;
    \\            row_max = new_max;
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    float inv_sum = 1.0f / row_sum;
    \\    uint out_offset = q_offset;
    \\    for (uint i = 0; i < 4; i++) {
    \\        uint d = simd_idx + i * 32;
    \\        if (d < head_dim) {
    \\            O[out_offset + d] = acc[i] * inv_sum;
    \\        }
    \\    }
    \\}
;
