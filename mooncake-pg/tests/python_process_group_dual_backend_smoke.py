"""Exercise the CUDA and CPU prototype groups in one rank process."""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.distributed as dist


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "mooncake_pg_python_probe", os.environ["MOONCAKE_PG_PYTHON_MODULE"]
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.register_cuda_backend("mooncake-python-cuda-dual")
    adapter.register_backend("mooncake-python-cpu-dual")
    adapter.MooncakePythonProcessGroup.set_host_ip(
        os.environ.get("MOONCAKE_CORE_HOST_IP", "127.0.0.1")
    )
    device_filters = os.environ.get("MOONCAKE_PGTEST_DEVICE_FILTERS", "")
    if device_filters:
        adapter.MooncakePythonProcessGroup.set_device_filter(
            [item for item in device_filters.split(",") if item]
        )

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    try:
        cuda_opts = adapter.MooncakeBackendOptions(
            torch.ones(world_size, dtype=torch.int32, device="cuda")
        )
        cpu_opts = adapter.MooncakeBackendOptions(
            torch.ones(world_size, dtype=torch.int32)
        )
        cuda_group = dist.new_group(
            backend="mooncake-python-cuda-dual", pg_options=cuda_opts
        )
        cpu_group = dist.new_group(
            backend="mooncake-python-cpu-dual", pg_options=cpu_opts
        )
        cuda_value = torch.tensor([rank + 1], device="cuda")
        cpu_value = torch.tensor([rank + 1])
        dist.all_reduce(cuda_value, group=cuda_group)
        dist.all_reduce(cpu_value, group=cpu_group)
        torch.cuda.synchronize()
        assert cuda_value.item() == 3 and cpu_value.item() == 3
        print(f"rank {rank}: dual prototype ProcessGroup smoke passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
