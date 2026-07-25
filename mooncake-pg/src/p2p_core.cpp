#include <p2p_core.h>

#include <cstring>
#include <cstdlib>
#include <mutex>

#include <cuda_alike.h>
#include <collective_reduce.h>
#include <memory_location.h>
#include <pg_core_check.h>
#include <pg_utils.h>

namespace mooncake {
namespace {

class CoreEngineRegistry {
   public:
    static CoreEngineRegistry& instance() {
        static auto* registry = new CoreEngineRegistry;
        return *registry;
    }

    TransferEngine* acquire(const std::string& requested_host_ip) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (external_engine_) return external_engine_;
        if (!owned_engine_) {
            owned_engine_ = std::make_unique<TransferEngine>(true);
            if (!device_filter_.empty()) {
                owned_engine_->setWhitelistFilters(
                    std::vector<std::string>(device_filter_));
            }
            const std::string& host_ip =
                host_ip_.empty() ? requested_host_ip : host_ip_;
            MOONCAKE_CORE_CHECK(
                owned_engine_->init(P2PHANDSHAKE, host_ip) == 0,
                "Failed to initialize the shared Mooncake core TransferEngine.");
        }
        return owned_engine_.get();
    }

    void setExternalEngine(TransferEngine* engine) {
        std::lock_guard<std::mutex> lock(mutex_);
        MOONCAKE_CORE_CHECK(!owned_engine_,
                            "set_transfer_engine must be called before "
                            "creating a Mooncake Python ProcessGroup.");
        external_engine_ = engine;
    }

    void setHostIp(std::string host_ip) {
        std::lock_guard<std::mutex> lock(mutex_);
        MOONCAKE_CORE_CHECK(!owned_engine_ && !external_engine_,
                            "set_host_ip must be called before creating a "
                            "Mooncake Python ProcessGroup.");
        host_ip_ = std::move(host_ip);
    }

    void setDeviceFilter(std::vector<std::string> filters) {
        std::lock_guard<std::mutex> lock(mutex_);
        MOONCAKE_CORE_CHECK(!owned_engine_ && !external_engine_,
                            "set_device_filter must be called before creating "
                            "a Mooncake Python ProcessGroup.");
        device_filter_ = std::move(filters);
    }

   private:
    std::mutex mutex_;
    std::unique_ptr<TransferEngine> owned_engine_;
    TransferEngine* external_engine_ = nullptr;
    std::string host_ip_;
    std::vector<std::string> device_filter_;
};

}  // namespace

bool CoreP2PWork::isCompleted() const {
    return status_->load(std::memory_order_acquire) != P2PProxy::OpStatus::kPending;
}

bool CoreP2PWork::isSuccess() const {
    return status_->load(std::memory_order_acquire) == P2PProxy::OpStatus::kSuccess;
}

bool CoreP2PWork::wait(std::chrono::milliseconds timeout) {
    BackoffWaiter waiter;
    const auto done = [this] { return isCompleted(); };
    const bool completed = timeout.count() > 0 ? waiter.wait_for(timeout, done)
                                                : (waiter.wait(done), true);
    MOONCAKE_CORE_CHECK(completed, "Mooncake P2P operation timed out.");
    MOONCAKE_CORE_CHECK(isSuccess(), "Mooncake P2P operation failed.");
    return true;
}

void MooncakeP2PCore::setExternalEngine(TransferEngine* engine) {
    CoreEngineRegistry::instance().setExternalEngine(engine);
}

void MooncakeP2PCore::setHostIp(std::string host_ip) {
    CoreEngineRegistry::instance().setHostIp(std::move(host_ip));
}

void MooncakeP2PCore::setDeviceFilter(std::vector<std::string> filters) {
    CoreEngineRegistry::instance().setDeviceFilter(std::move(filters));
}

std::string MooncakeP2PCore::getPreferredHca(const std::string& location) {
    static std::once_flag topology_once;
    static std::shared_ptr<Topology> topology;
    static TopologyMatrix matrix;
    std::call_once(topology_once, [] {
        auto* engine = CoreEngineRegistry::instance().acquire("127.0.0.1");
        topology = engine->getLocalTopology();
        if (topology) matrix = topology->getMatrix();
        if (!topology || matrix.empty()) {
            topology = std::make_shared<Topology>();
            topology->discover();
            matrix = topology->getMatrix();
        }
    });

    const auto it = matrix.find(location);
    if (it == matrix.end() || it->second.preferred_hca.empty()) return "";
    return it->second.preferred_hca.front();
}

MooncakeP2PCore::MooncakeP2PCore(int rank, int size,
                                 std::shared_ptr<CoreStore> store,
                                 std::string host_ip, int cuda_device_index,
                                 int backend_index)
    : rank_(rank), size_(size), cuda_device_index_(cuda_device_index),
      backend_index_(backend_index),
      store_(std::move(store)) {
    MOONCAKE_CORE_CHECK(static_cast<bool>(store_),
                        "MooncakeP2PCore requires a bootstrap store.");
    MOONCAKE_CORE_CHECK(rank_ >= 0 && rank_ < size_ && size_ > 0 &&
                            static_cast<size_t>(size_) <= kMaxNumRanks,
                        "Invalid P2P core rank or size.");

    engine_ = CoreEngineRegistry::instance().acquire(host_ip);
    local_server_name_ = engine_->getLocalIpAndPort();
    for (int i = 0; i < size_; ++i) local2global_rank_map_[i] = i;

    meta_ = std::make_shared<P2PConnectionMetadata>();
    meta_->rank = rank_;
    meta_->size = size_;
    meta_->activeSize = size_;
    meta_->taskCount = 0;
    meta_->engine = engine_;
    meta_->backendIndex = backend_index_;
    meta_->bufferBaseIndex = backend_index_ * 10;
    if (cuda_device_index_ < 0) {
        meta_->activeRanks = new bool[kMaxNumRanks]{};
    } else {
        MOONCAKE_CORE_CHECK(cudaHostAlloc(&meta_->activeRanks,
                                          kMaxNumRanks * sizeof(bool),
                                          cudaHostAllocMapped) == cudaSuccess,
                            "Failed to allocate CUDA active-rank bitmap.");
        MOONCAKE_CORE_CHECK(cudaHostGetDevicePointer(&meta_->activeRanksDevice,
                                                     meta_->activeRanks, 0) ==
                                cudaSuccess,
                            "Failed to map CUDA active-rank bitmap.");
    }
    for (int i = 0; i < size_; ++i) meta_->activeRanks[i] = true;

    allocateCollectiveBuffers();
    if (cuda_device_index_ >= 0) {
        cuda_collective_worker_ =
            CoreCudaCollectiveWorkerManager::getInstance().getCUDAWorker(
                cuda_device_index_);
    }

    const bool is_cpu = cuda_device_index_ < 0;
    p2p_worker_ = is_cpu
                      ? P2PDeviceWorkerManager::getInstance().getCPUWorker(engine_)
                      : P2PDeviceWorkerManager::getInstance().getCUDAWorker(
                            cuda_device_index_, engine_);
    p2p_proxy_ = std::make_shared<P2PProxy>(
        engine_, P2PProxy::Options{.is_cpu = is_cpu,
                                    .rank = rank_,
                                    .size = size_,
                                    .cuda_device_index = cuda_device_index_});
    p2p_proxy_->bindMeta(meta_);
    p2p_worker_->registerProxy(p2p_proxy_);

    connection_ctx_ = std::make_shared<ConnectionContext>(
        backend_index_, rank_, size_, false, local2global_rank_map_, store_, meta_,
        p2p_proxy_, engine_);

    SegmentInfo rank_info{};
    for (size_t i = 0; i < 2; ++i) {
        rank_info.send_buffer[i] =
            reinterpret_cast<uint64_t>(send_buffers_[i]);
        rank_info.recv_buffer[i] =
            reinterpret_cast<uint64_t>(recv_buffers_[i]);
        rank_info.send_sync[i] =
            reinterpret_cast<uint64_t>(sync_send_regions_[i]);
        rank_info.recv_sync[i] =
            reinterpret_cast<uint64_t>(sync_recv_regions_[i]);
    }
    rank_info.warmup_buffer[0] =
        reinterpret_cast<uint64_t>(connection_ctx_->warmup_send_region());
    rank_info.warmup_buffer[1] =
        reinterpret_cast<uint64_t>(connection_ctx_->warmup_recv_region());
    rank_info.p2p_credit_region =
        reinterpret_cast<uint64_t>(p2p_proxy_->credit_region());
    rank_info.p2p_ack_region = reinterpret_cast<uint64_t>(p2p_proxy_->ack_region());

    connection_ctx_->bootstrapLocalPeer(local_server_name_, rank_info);
    publishLocalPeerMetadata();
    ConnectionPoller::GetInstance().registerContext(connection_ctx_);
    connection_ctx_->waitUntilAllConnected();
}

MooncakeP2PCore::~MooncakeP2PCore() { shutdown(); }

void MooncakeP2PCore::allocateCollectiveBuffers() {
    const bool is_cpu = cuda_device_index_ < 0;
    const std::string location =
        is_cpu ? kWildcardLocation : GPU_PREFIX + std::to_string(cuda_device_index_);
    if (!is_cpu) {
        MOONCAKE_CORE_CHECK(cudaSetDevice(cuda_device_index_) == cudaSuccess,
                            "Failed to select CUDA device for PG core.");
    }
    for (size_t i = 0; i < 2; ++i) {
        if (is_cpu) {
            send_buffers_[i] = std::malloc(kBufferSize);
            recv_buffers_[i] = std::malloc(kBufferSize);
            MOONCAKE_CORE_CHECK(send_buffers_[i] && recv_buffers_[i],
                                "Failed to allocate CPU collective buffers.");
        } else {
            MOONCAKE_CORE_CHECK(cudaMalloc(&send_buffers_[i], kBufferSize) ==
                                    cudaSuccess &&
                                    cudaMalloc(&recv_buffers_[i], kBufferSize) ==
                                    cudaSuccess,
                                "Failed to allocate CUDA collective buffers.");
        }
        MOONCAKE_CORE_CHECK(
            !engine_->registerLocalMemory(send_buffers_[i], kBufferSize,
                                         location),
            "Failed to register collective send buffer.");
        MOONCAKE_CORE_CHECK(
            !engine_->registerLocalMemory(recv_buffers_[i], kBufferSize,
                                         location),
            "Failed to register collective receive buffer.");

        sync_send_regions_[i] = new int32_t[kMaxNumRanks]{};
        sync_recv_regions_[i] = new int32_t[kMaxNumRanks]{};
        MOONCAKE_CORE_CHECK(
            !engine_->registerLocalMemory(sync_send_regions_[i],
                                         kMaxNumRanks * sizeof(int32_t),
                                         kWildcardLocation),
            "Failed to register collective send sync region.");
        MOONCAKE_CORE_CHECK(
            !engine_->registerLocalMemory(sync_recv_regions_[i],
                                         kMaxNumRanks * sizeof(int32_t),
                                         kWildcardLocation),
            "Failed to register collective receive sync region.");
    }
}

void MooncakeP2PCore::releaseCollectiveBuffers() {
    const bool is_cpu = cuda_device_index_ < 0;
    for (size_t i = 0; i < 2; ++i) {
        if (sync_send_regions_[i]) {
            engine_->unregisterLocalMemory(sync_send_regions_[i]);
            delete[] sync_send_regions_[i];
            sync_send_regions_[i] = nullptr;
        }
        if (sync_recv_regions_[i]) {
            engine_->unregisterLocalMemory(sync_recv_regions_[i]);
            delete[] sync_recv_regions_[i];
            sync_recv_regions_[i] = nullptr;
        }
        if (send_buffers_[i]) {
            engine_->unregisterLocalMemory(send_buffers_[i]);
            if (is_cpu) std::free(send_buffers_[i]);
            else cudaFree(send_buffers_[i]);
            send_buffers_[i] = nullptr;
        }
        if (recv_buffers_[i]) {
            engine_->unregisterLocalMemory(recv_buffers_[i]);
            if (is_cpu) std::free(recv_buffers_[i]);
            else cudaFree(recv_buffers_[i]);
            recv_buffers_[i] = nullptr;
        }
    }
}

void MooncakeP2PCore::publishLocalPeerMetadata() {
    std::vector<uint8_t> rank_info_bytes(sizeof(SegmentInfo));
    std::memcpy(rank_info_bytes.data(), &meta_->segmentInfos[rank_],
                sizeof(SegmentInfo));
    store_->set(ConnectionContext::getBufferStoreKey(backend_index_, rank_),
                rank_info_bytes);
    store_->set(ConnectionContext::getServerNameStoreKey(backend_index_, rank_),
                local_server_name_);
}

void MooncakeP2PCore::waitForBatch(BatchID batch, size_t count) {
    BackoffWaiter waiter;
    TransferStatus status;
    waiter.wait([&] {
        for (size_t i = 0; i < count; ++i) {
            engine_->getTransferStatus(batch, i, status);
            MOONCAKE_CORE_CHECK(status.s != TransferStatusEnum::FAILED,
                                "Collective transfer failed.");
            if (status.s != TransferStatusEnum::COMPLETED) return false;
        }
        return true;
    });
    const auto result = engine_->freeBatchID(batch);
    MOONCAKE_CORE_CHECK(result.ok(), "Failed to release collective batch: ",
                        result.message());
}

void MooncakeP2PCore::synchronizeCollective(int buffer_offset) {
    auto* source = sync_send_regions_[buffer_offset];
    *source = 1;
    std::vector<TransferRequest> entries;
    for (int peer = 0; peer < size_; ++peer) {
        if (!meta_->activeRanks[peer]) continue;
        entries.push_back(TransferRequest{
            .opcode = TransferRequest::WRITE,
            .source = source,
            .target_id = meta_->segmentIDs[peer],
            .target_offset = meta_->segmentInfos[peer]
                                 .recv_sync[buffer_offset] +
                             rank_ * sizeof(int32_t),
            .length = sizeof(int32_t),
        });
    }
    const auto batch = engine_->allocateBatchID(entries.size());
    MOONCAKE_CORE_CHECK(engine_->submitTransfer(batch, entries).ok(),
                        "Failed to submit collective sync.");
    waitForBatch(batch, entries.size());

    auto* received = sync_recv_regions_[buffer_offset];
    BackoffWaiter waiter;
    waiter.wait([&] {
        for (int peer = 0; peer < size_; ++peer) {
            if (meta_->activeRanks[peer] && received[peer] != 1) return false;
        }
        return true;
    });
    for (int peer = 0; peer < size_; ++peer) received[peer] = 0;
}

void MooncakeP2PCore::allreduceCpu(RawBuffer input, RawBuffer output,
                                    ScalarType dtype, ReduceOp op) {
    collectiveCpu(CollectiveOp::kAllReduce, input, output, input.bytes, dtype,
                  op);
}

std::shared_ptr<CoreCudaCollectiveWork> MooncakeP2PCore::allreduceCuda(
    RawBuffer input, RawBuffer output, ScalarType dtype, ReduceOp op,
    cudaStream_t stream) {
    return collectiveCuda(CollectiveOp::kAllReduce, input, output, input.bytes,
                          dtype, op, 0, stream);
}

std::shared_ptr<CoreCudaCollectiveWork> MooncakeP2PCore::collectiveCuda(
    CollectiveOp collective_op, RawBuffer input, RawBuffer output,
    uint64_t unit_bytes, ScalarType dtype, ReduceOp reduce_op, int root_rank,
    cudaStream_t stream) {
    MOONCAKE_CORE_CHECK(!shutdown_, "MooncakeP2PCore is shut down.");
    MOONCAKE_CORE_CHECK(cuda_device_index_ >= 0,
                        "collectiveCuda requires a CUDA native core.");
    MOONCAKE_CORE_CHECK(root_rank >= 0 && root_rank < size_,
                        "Invalid collective root rank.");
    MOONCAKE_CORE_CHECK(cudaSetDevice(cuda_device_index_) == cudaSuccess,
                        "Failed to select CUDA device for collective.");

    const bool is_root = rank_ == root_rank;
    const bool needs_input = collective_op != CollectiveOp::kBarrier &&
        !((collective_op == CollectiveOp::kBroadcast ||
           collective_op == CollectiveOp::kScatter) && !is_root);
    const bool needs_output = collective_op != CollectiveOp::kBarrier &&
        !((collective_op == CollectiveOp::kReduce ||
           collective_op == CollectiveOp::kGather) && !is_root);
    if (needs_input) MOONCAKE_CORE_CHECK(input.data, "Collective input is required.");
    if (needs_output) MOONCAKE_CORE_CHECK(output.data, "Collective output is required.");
    const bool split_input = collective_op == CollectiveOp::kAllToAll ||
        collective_op == CollectiveOp::kReduceScatter ||
        collective_op == CollectiveOp::kScatter;
    const uint64_t input_bytes = split_input ? unit_bytes * size_ : unit_bytes;
    if (needs_input) MOONCAKE_CORE_CHECK(input.bytes >= input_bytes, "Collective input is too small.");
    const bool gathered_output = collective_op == CollectiveOp::kAllGather ||
        collective_op == CollectiveOp::kAllToAll || collective_op == CollectiveOp::kGather;
    if (needs_output) {
    MOONCAKE_CORE_CHECK(output.bytes >= (gathered_output ? unit_bytes * size_ : unit_bytes),
                            "Collective output is too small.");
    }

    return cuda_collective_worker_->enqueue(collective_op, input, output,
                                             unit_bytes, dtype, reduce_op,
                                             root_rank, meta_.get(), stream);
}

void MooncakeP2PCore::collectiveCpu(CollectiveOp collective_op,
                                    RawBuffer input, RawBuffer output,
                                    uint64_t unit_bytes, ScalarType dtype,
                                    ReduceOp reduce_op, int root_rank) {
    MOONCAKE_CORE_CHECK(!shutdown_, "MooncakeP2PCore is shut down.");
    MOONCAKE_CORE_CHECK(cuda_device_index_ < 0,
                        "collectiveCpu requires a CPU native core.");
    MOONCAKE_CORE_CHECK(root_rank >= 0 && root_rank < size_,
                        "Invalid collective root rank.");
    MOONCAKE_CORE_CHECK(unit_bytes <= kBufferSize / size_,
                        "Collective unit exceeds the native buffer.");

    const bool is_root = rank_ == root_rank;
    const bool needs_input =
        collective_op != CollectiveOp::kBarrier &&
        !((collective_op == CollectiveOp::kBroadcast ||
           collective_op == CollectiveOp::kScatter) &&
          !is_root);
    const bool needs_output =
        collective_op != CollectiveOp::kBarrier &&
        !( (collective_op == CollectiveOp::kReduce ||
           collective_op == CollectiveOp::kGather) && !is_root);
    if (needs_input) {
        MOONCAKE_CORE_CHECK(input.data, "Collective input is required.");
    }
    if (needs_output) {
        MOONCAKE_CORE_CHECK(output.data, "Collective output is required.");
    }

    const bool split_input = collective_op == CollectiveOp::kAllToAll ||
                             collective_op == CollectiveOp::kReduceScatter ||
                             collective_op == CollectiveOp::kScatter;
    const uint64_t input_bytes = split_input ? unit_bytes * size_ : unit_bytes;
    if (needs_input) {
        MOONCAKE_CORE_CHECK(input.bytes >= input_bytes,
                            "Collective input is too small.");
    }
    const bool gathered_output = collective_op == CollectiveOp::kAllGather ||
                                collective_op == CollectiveOp::kAllToAll ||
                                collective_op == CollectiveOp::kGather;
    if (needs_output) {
        const uint64_t output_bytes = gathered_output ? unit_bytes * size_ : unit_bytes;
        MOONCAKE_CORE_CHECK(output.bytes >= output_bytes,
                            "Collective output is too small.");
    }

    const int buffer_offset = meta_->taskCount++ % 2;
    if (needs_input) {
        std::memcpy(send_buffers_[buffer_offset], input.data, input_bytes);
    }
    std::vector<TransferRequest> entries;
    for (int peer = 0; peer < size_; ++peer) {
        if (!meta_->activeRanks[peer]) continue;
        if (collective_op == CollectiveOp::kBarrier ||
            ((collective_op == CollectiveOp::kBroadcast ||
              collective_op == CollectiveOp::kScatter) &&
             !is_root) ||
            ((collective_op == CollectiveOp::kGather ||
              collective_op == CollectiveOp::kReduce) &&
             peer != root_rank)) {
            continue;
        }
        auto* source = static_cast<char*>(send_buffers_[buffer_offset]);
        if (split_input) source += peer * unit_bytes;
        uint64_t target_offset = meta_->segmentInfos[peer]
                                     .recv_buffer[buffer_offset];
        if (collective_op != CollectiveOp::kBroadcast &&
            collective_op != CollectiveOp::kScatter) {
            target_offset += rank_ * unit_bytes;
        }
        entries.push_back(TransferRequest{
            .opcode = TransferRequest::WRITE,
            .source = source,
            .target_id = meta_->segmentIDs[peer],
            .target_offset = target_offset,
            .length = unit_bytes,
        });
    }
    if (!entries.empty()) {
        const auto batch = engine_->allocateBatchID(entries.size());
        MOONCAKE_CORE_CHECK(engine_->submitTransfer(batch, entries).ok(),
                            "Failed to submit native collective.");
        waitForBatch(batch, entries.size());
    }
    synchronizeCollective(buffer_offset);

    const auto* received = recv_buffers_[buffer_offset];
    switch (collective_op) {
        case CollectiveOp::kAllReduce:
        case CollectiveOp::kReduceScatter:
            reduceRawCpu(output.data, received, unit_bytes, dtype, size_,
                         reduce_op, meta_->activeRanks);
            break;
        case CollectiveOp::kReduce:
            if (is_root) {
                reduceRawCpu(output.data, received, unit_bytes, dtype, size_,
                             reduce_op, meta_->activeRanks);
            }
            break;
        case CollectiveOp::kAllGather:
        case CollectiveOp::kAllToAll:
            std::memcpy(output.data, received, unit_bytes * size_);
            break;
        case CollectiveOp::kGather:
            if (is_root) std::memcpy(output.data, received, unit_bytes * size_);
            break;
        case CollectiveOp::kBroadcast:
        case CollectiveOp::kScatter:
            std::memcpy(output.data, received, unit_bytes);
            break;
        case CollectiveOp::kBarrier: break;
    }
}

std::shared_ptr<CoreP2PWork> MooncakeP2PCore::send(RawBuffer buffer,
                                                     int dst_rank,
                                                     cudaStream_t stream) {
    MOONCAKE_CORE_CHECK(!shutdown_, "MooncakeP2PCore is shut down.");
    MOONCAKE_CORE_CHECK(dst_rank >= 0 && dst_rank < size_, "Invalid dst rank.");
    auto status = std::make_shared<std::atomic<P2PProxy::OpStatus>>(
        P2PProxy::OpStatus::kPending);
    p2p_proxy_->enqueueSend(P2PProxy::SendOp{
        .buffer_ = buffer,
        .peer_rank_ = dst_rank,
        .cuda_stream_ = stream,
        .status_ = status,
        .keepalive_ = {},
    });
    return std::make_shared<CoreP2PWork>(std::move(status));
}

std::shared_ptr<CoreP2PWork> MooncakeP2PCore::recv(RawBuffer buffer,
                                                     int src_rank,
                                                     cudaStream_t stream) {
    MOONCAKE_CORE_CHECK(!shutdown_, "MooncakeP2PCore is shut down.");
    MOONCAKE_CORE_CHECK(src_rank >= 0 && src_rank < size_, "Invalid src rank.");
    auto status = std::make_shared<std::atomic<P2PProxy::OpStatus>>(
        P2PProxy::OpStatus::kPending);
    p2p_proxy_->enqueueRecv(P2PProxy::RecvOp{
        .buffer_ = buffer,
        .peer_rank_ = src_rank,
        .cuda_stream_ = stream,
        .status_ = status,
        .keepalive_ = {},
        .completion_callback_ = {},
    });
    return std::make_shared<CoreP2PWork>(std::move(status));
}

void MooncakeP2PCore::shutdown() {
    if (shutdown_) return;
    shutdown_ = true;

    bool hung = false;
    if (p2p_worker_ && p2p_proxy_) {
        p2p_worker_->removeProxy(p2p_proxy_);
        hung |= !p2p_proxy_->drainTasks();
    }
    if (connection_ctx_) {
        connection_ctx_->shutdown();
        ConnectionPoller::GetInstance().removeContext(connection_ctx_);
        hung |= !connection_ctx_->drainPoller();
    }
    if (hung && p2p_proxy_) p2p_proxy_->abandonResources();

    connection_ctx_.reset();
    p2p_proxy_.reset();
    p2p_worker_.reset();
    if (!hung) releaseCollectiveBuffers();
    if (cuda_device_index_ < 0) delete[] meta_->activeRanks;
    else cudaFreeHost(meta_->activeRanks);
    meta_->activeRanks = nullptr;
    meta_.reset();
}

}  // namespace mooncake
