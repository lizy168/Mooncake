#include <collective_reduce.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>

namespace mooncake {
namespace {

template <typename T>
T apply(T lhs, T rhs, ReduceOp op) {
    switch (op) {
        case ReduceOp::kSum: return lhs + rhs;
        case ReduceOp::kProduct: return lhs * rhs;
        case ReduceOp::kMin: return std::min(lhs, rhs);
        case ReduceOp::kMax: return std::max(lhs, rhs);
    }
    throw std::runtime_error("Unsupported reduce operation.");
}

template <>
bool apply(bool lhs, bool rhs, ReduceOp op) {
    switch (op) {
        case ReduceOp::kSum:
            return lhs || rhs;
        case ReduceOp::kProduct:
            return lhs && rhs;
        case ReduceOp::kMin:
            return lhs && rhs;
        case ReduceOp::kMax:
            return lhs || rhs;
    }
    throw std::runtime_error("Unsupported reduce operation.");
}

template <typename T>
void reduce(void* dst, const void* src, size_t bytes, size_t num_ranks,
            ReduceOp op, const bool* active_ranks) {
    auto* output = static_cast<T*>(dst);
    const auto* input = static_cast<const T*>(src);
    const size_t count = bytes / sizeof(T);
    for (size_t i = 0; i < count; ++i) {
        bool initialized = false;
        T value{};
        for (size_t rank = 0; rank < num_ranks; ++rank) {
            if (!active_ranks[rank]) continue;
            const T next = input[rank * count + i];
            value = initialized ? apply(value, next, op) : next;
            initialized = true;
        }
        output[i] = value;
    }
}

float bfloat16ToFloat(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

uint16_t floatToBfloat16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    // Round to nearest even before truncating the IEEE float payload.
    bits += 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

void reduceBfloat16(void* dst, const void* src, size_t bytes,
                    size_t num_ranks, ReduceOp op, const bool* active_ranks) {
    auto* output = static_cast<uint16_t*>(dst);
    const auto* input = static_cast<const uint16_t*>(src);
    const size_t count = bytes / sizeof(uint16_t);
    for (size_t i = 0; i < count; ++i) {
        bool initialized = false;
        float value{};
        for (size_t rank = 0; rank < num_ranks; ++rank) {
            if (!active_ranks[rank]) continue;
            const float next = bfloat16ToFloat(input[rank * count + i]);
            value = initialized ? apply(value, next, op) : next;
            initialized = true;
        }
        output[i] = floatToBfloat16(value);
    }
}

}  // namespace

void reduceRawCpu(void* dst, const void* src, size_t bytes, ScalarType dtype,
                  size_t num_ranks, ReduceOp op, const bool* active_ranks) {
    switch (dtype) {
        case ScalarType::kUInt8: return reduce<uint8_t>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kInt8: return reduce<int8_t>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kInt16: return reduce<int16_t>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kInt32: return reduce<int32_t>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kInt64: return reduce<int64_t>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kFloat32: return reduce<float>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kFloat64: return reduce<double>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kBool: return reduce<bool>(dst, src, bytes, num_ranks, op, active_ranks);
        case ScalarType::kBFloat16:
            return reduceBfloat16(dst, src, bytes, num_ranks, op, active_ranks);
    }
    throw std::runtime_error("Unsupported raw CPU reduce dtype.");
}

}  // namespace mooncake
