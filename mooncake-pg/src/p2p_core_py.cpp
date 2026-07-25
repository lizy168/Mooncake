#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <p2p_core.h>

namespace py = pybind11;

namespace mooncake {
namespace {

class PythonCoreStore final : public CoreStore {
   public:
    explicit PythonCoreStore(py::object store) : store_(std::move(store)) {}

    bool check(const std::vector<std::string>& keys) override {
        py::gil_scoped_acquire gil;
        return store_.attr("check")(keys).cast<bool>();
    }

    std::vector<uint8_t> get(const std::string& key) override {
        py::gil_scoped_acquire gil;
        const std::string value = store_.attr("get")(key).cast<py::bytes>();
        return {value.begin(), value.end()};
    }

    std::string getString(const std::string& key) override {
        py::gil_scoped_acquire gil;
        return store_.attr("get")(key).cast<py::bytes>();
    }

    void set(const std::string& key,
             const std::vector<uint8_t>& value) override {
        py::gil_scoped_acquire gil;
        store_.attr("set")(key, py::bytes(
            reinterpret_cast<const char*>(value.data()), value.size()));
    }

    void set(const std::string& key, const std::string& value) override {
        py::gil_scoped_acquire gil;
        store_.attr("set")(key, value);
    }

    void deleteKey(const std::string& key) override {
        py::gil_scoped_acquire gil;
        store_.attr("delete_key")(key);
    }

   private:
    py::object store_;
};

}  // namespace

PYBIND11_MODULE(pg_core, m) {
    m.def("set_transfer_engine", [](uintptr_t engine) {
        MooncakeP2PCore::setExternalEngine(
            reinterpret_cast<TransferEngine*>(engine));
    }, py::arg("engine") = 0,
       "Inject an initialized non-owning TransferEngine before PG creation.");
    m.def("set_host_ip", &MooncakeP2PCore::setHostIp, py::arg("host_ip"));
    m.def("set_device_filter", &MooncakeP2PCore::setDeviceFilter,
          py::arg("filters"));
    m.def("get_preferred_hca", &MooncakeP2PCore::getPreferredHca,
          py::arg("location"));

    py::class_<CoreP2PWork, std::shared_ptr<CoreP2PWork>>(m, "P2PWork")
        .def("is_completed", &CoreP2PWork::isCompleted)
        .def("is_success", &CoreP2PWork::isSuccess)
        .def("wait", [](CoreP2PWork& work, int timeout_ms) {
            return work.wait(std::chrono::milliseconds(timeout_ms));
        }, py::arg("timeout_ms") = 0);

    py::class_<CoreCudaCollectiveWork, std::shared_ptr<CoreCudaCollectiveWork>>(
        m, "CudaCollectiveWork")
        .def("is_completed", &CoreCudaCollectiveWork::isCompleted)
        .def("wait", [](const CoreCudaCollectiveWork& work, int timeout_ms,
                         uintptr_t cuda_stream) {
            return work.wait(reinterpret_cast<cudaStream_t>(cuda_stream),
                             std::chrono::milliseconds(timeout_ms));
        }, py::arg("timeout_ms") = 0, py::arg("cuda_stream") = 0);

    py::class_<MooncakeP2PCore>(m, "P2PCore")
        .def(py::init([](int rank, int size, py::object store,
                         const std::string& host_ip, int cuda_device_index,
                         int backend_index) {
            auto core_store =
                std::make_shared<PythonCoreStore>(std::move(store));
            // The native constructor waits for ConnectionPoller, whose
            // CoreStore callbacks reacquire the GIL on its worker thread.
            py::gil_scoped_release release;
            return std::make_unique<MooncakeP2PCore>(
                rank, size, std::move(core_store), host_ip, cuda_device_index,
                backend_index);
        }), py::arg("rank"), py::arg("size"), py::arg("store"),
            py::arg("host_ip") = "127.0.0.1",
            py::arg("cuda_device_index") = -1, py::arg("backend_index") = 1)
        .def("send", [](MooncakeP2PCore& core, uintptr_t data, uint64_t bytes,
                         int dst_rank, uintptr_t cuda_stream) {
            return core.send(RawBuffer{reinterpret_cast<void*>(data), bytes},
                             dst_rank,
                             reinterpret_cast<cudaStream_t>(cuda_stream));
        }, py::arg("data"), py::arg("bytes"), py::arg("dst_rank"),
           py::arg("cuda_stream") = 0)
        .def("recv", [](MooncakeP2PCore& core, uintptr_t data, uint64_t bytes,
                         int src_rank, uintptr_t cuda_stream) {
            return core.recv(RawBuffer{reinterpret_cast<void*>(data), bytes},
                             src_rank,
                             reinterpret_cast<cudaStream_t>(cuda_stream));
        }, py::arg("data"), py::arg("bytes"), py::arg("src_rank"),
           py::arg("cuda_stream") = 0)
        .def("allreduce_cpu", [](MooncakeP2PCore& core, uintptr_t input,
                                  uintptr_t output, uint64_t bytes, int dtype,
                                  int reduce_op) {
            py::gil_scoped_release release;
            core.allreduceCpu(
                RawBuffer{reinterpret_cast<void*>(input), bytes},
                RawBuffer{reinterpret_cast<void*>(output), bytes},
                static_cast<ScalarType>(dtype),
                static_cast<ReduceOp>(reduce_op));
        }, py::arg("input"), py::arg("output"), py::arg("bytes"),
           py::arg("dtype"), py::arg("reduce_op"))
        .def("allreduce_cuda", [](MooncakeP2PCore& core, uintptr_t input,
                                   uintptr_t output, uint64_t bytes, int dtype,
                                   int reduce_op, uintptr_t stream) {
            py::gil_scoped_release release;
            return core.allreduceCuda(
                RawBuffer{reinterpret_cast<void*>(input), bytes},
                RawBuffer{reinterpret_cast<void*>(output), bytes},
                static_cast<ScalarType>(dtype),
                static_cast<ReduceOp>(reduce_op),
                reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("input"), py::arg("output"), py::arg("bytes"),
           py::arg("dtype"), py::arg("reduce_op"), py::arg("stream"))
        .def("collective_cpu", [](MooncakeP2PCore& core, int collective_op,
                                   uintptr_t input, uint64_t input_bytes,
                                   uintptr_t output, uint64_t output_bytes,
                                   uint64_t unit_bytes, int dtype,
                                   int reduce_op, int root_rank) {
            py::gil_scoped_release release;
            core.collectiveCpu(
                static_cast<CollectiveOp>(collective_op),
                RawBuffer{reinterpret_cast<void*>(input), input_bytes},
                RawBuffer{reinterpret_cast<void*>(output), output_bytes},
                unit_bytes, static_cast<ScalarType>(dtype),
                static_cast<ReduceOp>(reduce_op), root_rank);
        }, py::arg("collective_op"), py::arg("input") = 0,
           py::arg("input_bytes") = 0, py::arg("output") = 0,
           py::arg("output_bytes") = 0, py::arg("unit_bytes") = 0,
           py::arg("dtype") = 0, py::arg("reduce_op") = 0,
           py::arg("root_rank") = 0)
        .def("collective_cuda", [](MooncakeP2PCore& core, int collective_op,
                                    uintptr_t input, uint64_t input_bytes,
                                    uintptr_t output, uint64_t output_bytes,
                                    uint64_t unit_bytes, int dtype,
                                    int reduce_op, int root_rank,
                                    uintptr_t stream) {
            py::gil_scoped_release release;
            return core.collectiveCuda(
                static_cast<CollectiveOp>(collective_op),
                RawBuffer{reinterpret_cast<void*>(input), input_bytes},
                RawBuffer{reinterpret_cast<void*>(output), output_bytes},
                unit_bytes, static_cast<ScalarType>(dtype),
                static_cast<ReduceOp>(reduce_op), root_rank,
                reinterpret_cast<cudaStream_t>(stream));
        }, py::arg("collective_op"), py::arg("input") = 0,
           py::arg("input_bytes") = 0, py::arg("output") = 0,
           py::arg("output_bytes") = 0, py::arg("unit_bytes") = 0,
           py::arg("dtype") = 0, py::arg("reduce_op") = 0,
           py::arg("root_rank") = 0, py::arg("stream") = 0)
        .def("shutdown", &MooncakeP2PCore::shutdown);
}

}  // namespace mooncake
