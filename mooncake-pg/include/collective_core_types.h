#ifndef MOONCAKE_COLLECTIVE_CORE_TYPES_H
#define MOONCAKE_COLLECTIVE_CORE_TYPES_H

#include <cstddef>
#include <cstdint>

#include <pg_core_types.h>
#include <transfer_engine.h>

namespace mooncake {

// Shared-memory task record consumed by both the host worker and the CUDA
// enqueue kernel. Keep this ABI independent of framework operation enums.
struct Task {
    volatile bool active = false;
    CollectiveOp op = CollectiveOp::kBarrier;
    size_t tensorSize = 0;
    int64_t broadcastRoot = 0;
    int bufferOffset = 0;
    uint64_t submitSequence = 0;
    BatchID batchID{};
    void* transferGroupMeta = nullptr;
};

}  // namespace mooncake

#endif  // MOONCAKE_COLLECTIVE_CORE_TYPES_H
