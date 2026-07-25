#include <collective_reduce.h>
#include <pg_core_check.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mooncake {
namespace {

template <typename T>
__device__ T apply(T lhs, T rhs, ReduceOp op) {
    switch (op) {
        case ReduceOp::kSum: return lhs + rhs;
        case ReduceOp::kProduct: return lhs * rhs;
        case ReduceOp::kMin: return lhs < rhs ? lhs : rhs;
        case ReduceOp::kMax: return lhs > rhs ? lhs : rhs;
    }
    return lhs;
}

template <typename T>
__global__ void reduceKernel(T* output, const T* input, size_t count,
                             size_t ranks, ReduceOp op,
                             const bool* active_ranks) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        bool initialized = false;
        T value{};
        for (size_t rank = 0; rank < ranks; ++rank) {
            if (!active_ranks[rank]) continue;
            const T next = input[rank * count + i];
            value = initialized ? apply(value, next, op) : next;
            initialized = true;
        }
        output[i] = value;
    }
}

__global__ void reduceBfloat16Kernel(__nv_bfloat16* output,
                                     const __nv_bfloat16* input,
                                     size_t count, size_t ranks, ReduceOp op,
                                     const bool* active_ranks) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        bool initialized = false;
        float value{};
        for (size_t rank = 0; rank < ranks; ++rank) {
            if (!active_ranks[rank]) continue;
            const float next = __bfloat162float(input[rank * count + i]);
            switch (op) {
                case ReduceOp::kSum: value = initialized ? value + next : next; break;
                case ReduceOp::kProduct: value = initialized ? value * next : next; break;
                case ReduceOp::kMin: value = initialized ? fminf(value, next) : next; break;
                case ReduceOp::kMax: value = initialized ? fmaxf(value, next) : next; break;
            }
            initialized = true;
        }
        output[i] = __float2bfloat16(value);
    }
}

template <typename T>
void launch(void* dst, const void* src, size_t bytes, size_t ranks,
            ReduceOp op, const bool* active_ranks, cudaStream_t stream) {
    const size_t count = bytes / sizeof(T);
    reduceKernel<<<64, 256, 0, stream>>>(static_cast<T*>(dst),
        static_cast<const T*>(src), count, ranks, op, active_ranks);
}

}  // namespace

void reduceRawCuda(void* dst, const void* src, size_t bytes, ScalarType dtype,
                   size_t num_ranks, ReduceOp op, const bool* active_ranks,
                   cudaStream_t stream) {
    switch (dtype) {
        case ScalarType::kUInt8: launch<uint8_t>(dst, src, bytes, num_ranks, op, active_ranks, stream); break;
        case ScalarType::kInt8: launch<int8_t>(dst, src, bytes, num_ranks, op, active_ranks, stream); break;
        case ScalarType::kInt16: launch<int16_t>(dst, src, bytes, num_ranks, op, active_ranks, stream); break;
        case ScalarType::kInt32: launch<int32_t>(dst, src, bytes, num_ranks, op, active_ranks, stream); break;
        case ScalarType::kInt64: launch<int64_t>(dst, src, bytes, num_ranks, op, active_ranks, stream); break;
        case ScalarType::kFloat32: launch<float>(dst, src, bytes, num_ranks, op, active_ranks, stream); break;
        case ScalarType::kFloat64: launch<double>(dst, src, bytes, num_ranks, op, active_ranks, stream); break;
        case ScalarType::kBFloat16:
            reduceBfloat16Kernel<<<64, 256, 0, stream>>>(
                static_cast<__nv_bfloat16*>(dst),
                static_cast<const __nv_bfloat16*>(src), bytes / 2, num_ranks,
                op, active_ranks);
            break;
        case ScalarType::kBool:
            MOONCAKE_CORE_CHECK(false, "CUDA bool reduction is unsupported.");
    }
    MOONCAKE_CORE_CHECK(cudaGetLastError() == cudaSuccess,
                        "Failed to launch CUDA raw reduction.");
}

}  // namespace mooncake
