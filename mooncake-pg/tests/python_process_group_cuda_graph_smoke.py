"""Verify the experimental CUDA ProcessGroup can be called during capture."""

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
    adapter.register_cuda_backend("mooncake-python-cuda-graph-smoke")

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    options = adapter.MooncakeBackendOptions(
        torch.zeros(world_size, dtype=torch.int32, device="cuda")
    )
    dist.init_process_group(
        backend="mooncake-python-cuda-graph-smoke", pg_options=options
    )
    try:
        static_input = torch.full((1,), rank + 1, dtype=torch.int32, device="cuda")
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            dist.all_reduce(static_input)
        graph.replay()
        torch.cuda.synchronize()
        assert static_input.item() == 3
        print(f"rank {rank}: native CUDA graph ProcessGroup smoke passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
