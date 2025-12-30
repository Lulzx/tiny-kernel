# tiny-kernel

metal kernels for llm inference. no python. no pytorch. just zig and metal.

## why

mlx is slow. pytorch mps is slower. nvidia has cuda. we have this.

fused kernels = less memory bandwidth = faster inference. it's not complicated.

## what's in the box

| kernel | what it does |
|--------|-------------|
| `rmsnorm` | two-level simd reduction. 32 lanes go brr |
| `rope` | rotary embeddings. precomputed cos/sin |
| `swiglu` | silu(gate) * up. the activation llama uses |
| `fused_rmsnorm_rope_swiglu` | all three in one kernel. zero intermediate tensors |
| `flash_attention_fwd` | online softmax. O(1) memory. causal mask |
| `paged_attention` | kv cache with block tables. for real inference |
| `gqa_attention` | grouped query attention. fewer kv heads |
| `qmatmul_int4` | int4 with fused dequant. 4x less memory |

## build

```
zig build
```

## run benchmarks

```
zig build run
```

```
tiny-kernel benchmarks
Device: Apple M4 Pro
========================

RMSNorm [32x4096]: 0.205 ms/iter
Fused RMSNorm+RoPE+SwiGLU [1x4096->14336]: 82.238 ms/iter
Flash Attention [heads=32, seq=512, dim=128]: 6.213 ms/iter

Done.
```

## the code

~600 lines of zig. ~400 lines of metal. no comments because the code is the comment.

```
src/
├── main.zig           # exports
├── bench.zig          # benchmarks
├── metal/
│   ├── objc.zig       # objc runtime. msgSend is all you need
│   ├── device.zig     # MTLDevice wrapper
│   ├── buffer.zig     # typed gpu buffers
│   └── encoder.zig    # command encoder
└── kernels/
    ├── shaders.zig    # all the metal. embedded strings
    ├── rmsnorm.zig
    ├── rope.zig
    ├── swiglu.zig
    ├── fused.zig
    ├── attention.zig
    └── quantized.zig
```

## metal tips for the uninitiated

- simd width is 32. always. don't check, just hardcode it
- `simd_sum` is free. use it
- threadgroup size 256 = 8 simd groups. good occupancy
- avoid atomics. they're emulated. use two-level reduction instead
- `threadgroup` memory is fast. global memory is slow. plan accordingly

## fusion strategy

standard llama ffn:
1. rmsnorm: read x, write normed (memory bound)
2. gate proj: read normed, write gate (compute bound)
3. up proj: read normed, write up (compute bound)
4. silu: read gate, write activated (memory bound)
5. mul: read activated + up, write out (memory bound)

fused kernel:
1. read x + all weights, write out

5 kernel launches -> 1. that's the whole trick.

## philosophy

- if you can fuse it, fuse it
- memory bandwidth is the enemy
- the best kernel is the one that doesn't exist
- complexity is debt. pay it down

## requirements

- zig 0.15+
- macos with metal
- apple silicon (m1/m2/m3/m4)

## license

WTFPL. do what the fuck you want. just ship something.
