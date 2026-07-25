#ifndef MOONCAKE_P2P_CORE_H
#define MOONCAKE_P2P_CORE_H

#include <atomic>
#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <connection_poller.h>
#include <core_cuda_collective_worker.h>
#include <p2p_proxy.h>
#include <pg_core_store.h>

namespace mooncake {

class CoreP2PWork {
   public:
    explicit CoreP2PWork(std::shared_ptr<std::atomic<P2PProxy::OpStatus>> status)
        : status_(std::move(status)) {}

    bool isCompleted() const;
    bool isSuccess() const;
    bool wait(std::chrono::milliseconds timeout = std::chrono::milliseconds(0));

   private:
    std::shared_ptr<std::atomic<P2PProxy::OpStatus>> status_;
};

// CPU-only native core used to prove that P2P does not need a Torch C++ PG.
// The Python adapter owns tensor validation and lifetime; this class only sees
// raw address/length descriptors and a framework-neutral bootstrap store.
class MooncakeP2PCore {
   public:
    MooncakeP2PCore(int rank, int size, std::shared_ptr<CoreStore> store,
                    std::string host_ip = "127.0.0.1",
                    int cuda_device_index = -1, int backend_index = 1);
    ~MooncakeP2PCore();

    MooncakeP2PCore(const MooncakeP2PCore&) = delete;
    MooncakeP2PCore& operator=(const MooncakeP2PCore&) = delete;

    // Process-wide configuration mirroring the public PG controls. The core
    // stays framework-neutral: callers provide only a native engine handle.
    static void setExternalEngine(TransferEngine* engine);
    static void setHostIp(std::string host_ip);
    static void setDeviceFilter(std::vector<std::string> filters);
    static std::string getPreferredHca(const std::string& location);

    std::shared_ptr<CoreP2PWork> send(RawBuffer buffer, int dst_rank,
                                      cudaStream_t stream = nullptr);
    std::shared_ptr<CoreP2PWork> recv(RawBuffer buffer, int src_rank,
                                      cudaStream_t stream = nullptr);
    void allreduceCpu(RawBuffer input, RawBuffer output, ScalarType dtype,
                      ReduceOp op);
    void collectiveCpu(CollectiveOp collective_op, RawBuffer input,
                       RawBuffer output, uint64_t unit_bytes,
                       ScalarType dtype = ScalarType::kUInt8,
                       ReduceOp reduce_op = ReduceOp::kSum,
                       int root_rank = 0);
    std::shared_ptr<CoreCudaCollectiveWork> allreduceCuda(
        RawBuffer input, RawBuffer output, ScalarType dtype, ReduceOp op,
        cudaStream_t stream);
    std::shared_ptr<CoreCudaCollectiveWork> collectiveCuda(
        CollectiveOp collective_op, RawBuffer input, RawBuffer output,
        uint64_t unit_bytes, ScalarType dtype, ReduceOp reduce_op,
        int root_rank, cudaStream_t stream);
    void shutdown();

   private:
    void allocateCollectiveBuffers();
    void releaseCollectiveBuffers();
    void waitForBatch(BatchID batch, size_t count);
    void synchronizeCollective(int buffer_offset);
    void publishLocalPeerMetadata();

    int rank_;
    int size_;
    int cuda_device_index_;
    int backend_index_;
    // Owned by the process-wide core engine registry. It either owns one
    // default engine or borrows the injected Python TransferEngine.
    TransferEngine* engine_ = nullptr;
    std::shared_ptr<CoreStore> store_;
    std::shared_ptr<P2PConnectionMetadata> meta_;
    std::shared_ptr<P2PProxy> p2p_proxy_;
    std::shared_ptr<P2PDeviceWorker> p2p_worker_;
    std::shared_ptr<CoreCudaCollectiveWorker> cuda_collective_worker_;
    std::shared_ptr<ConnectionContext> connection_ctx_;
    std::array<void*, 2> send_buffers_{};
    std::array<void*, 2> recv_buffers_{};
    std::array<int32_t*, 2> sync_send_regions_{};
    std::array<int32_t*, 2> sync_recv_regions_{};
    uint64_t local2global_rank_map_[kMaxNumRanks]{};
    std::string local_server_name_;
    bool shutdown_{false};
};

}  // namespace mooncake

#endif  // MOONCAKE_P2P_CORE_H
