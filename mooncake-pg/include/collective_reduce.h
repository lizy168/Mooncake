#ifndef MOONCAKE_COLLECTIVE_REDUCE_H
#define MOONCAKE_COLLECTIVE_REDUCE_H

#include <cstddef>

#include <cuda_alike.h>
#include <pg_core_types.h>

namespace mooncake {

void reduceRawCpu(void* dst, const void* src, size_t bytes, ScalarType dtype,
                  size_t num_ranks, ReduceOp op, const bool* active_ranks);

void reduceRawCuda(void* dst, const void* src, size_t bytes, ScalarType dtype,
                   size_t num_ranks, ReduceOp op, const bool* active_ranks,
                   cudaStream_t stream);

}  // namespace mooncake

#endif  // MOONCAKE_COLLECTIVE_REDUCE_H
