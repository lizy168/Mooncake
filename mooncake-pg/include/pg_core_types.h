#ifndef MOONCAKE_PG_CORE_TYPES_H
#define MOONCAKE_PG_CORE_TYPES_H

#include <cstddef>
#include <cstdint>
#include <functional>

#include <transfer_engine.h>

namespace mooncake {

inline constexpr size_t kBufferSize = 1u << 24;
inline constexpr size_t kMaxNumRanks = 64;

struct SegmentInfo {
    uint64_t send_buffer[2], recv_buffer[2], send_sync[2], recv_sync[2],
        warmup_buffer[2];
    uint64_t p2p_credit_region;
    uint64_t p2p_ack_region;
};

// Communication state shared by the connection layer and the raw P2P core.
// Framework adapters own any tensor or store state outside this structure.
struct P2PConnectionMetadata {
    int rank;
    int size;
    int activeSize;
    int taskCount;
    bool* activeRanks;
    bool* activeRanksDevice;
    bool peerConnected[kMaxNumRanks]{};
    TransferEngine* engine;
    int bufferBaseIndex;
    int backendIndex;
    TransferMetadata::SegmentID segmentIDs[kMaxNumRanks];
    SegmentInfo segmentInfos[kMaxNumRanks];
    std::function<void(int)> onPeerBroken;
};

struct RawBuffer {
    void* data = nullptr;
    uint64_t bytes = 0;
};

// Stable operation descriptors shared by the native core and framework
// adapters.  Their values intentionally remain independent of c10d enums.
enum class CollectiveOp : int {
    kBroadcast,
    kAllReduce,
    kAllGather,
    kReduceScatter,
    kAllToAll,
    kReduce,
    kGather,
    kScatter,
    kBarrier,
};

enum class ScalarType : int {
    kUInt8,
    kInt8,
    kInt16,
    kInt32,
    kInt64,
    kFloat32,
    kFloat64,
    kBool,
    kBFloat16,
};

enum class ReduceOp : int { kSum, kProduct, kMin, kMax };

}  // namespace mooncake

#endif  // MOONCAKE_PG_CORE_TYPES_H
