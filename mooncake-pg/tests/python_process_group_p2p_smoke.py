"""Two-rank P2P smoke for the experimental Python Mooncake ProcessGroup."""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.distributed as dist


def load_adapter():
    adapter_path = os.environ["MOONCAKE_PG_PYTHON_MODULE"]
    module_spec = importlib.util.spec_from_file_location(
        "mooncake_pg_python_probe", adapter_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load adapter from {adapter_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def main() -> None:
    adapter = load_adapter()
    adapter.register_backend("mooncake-python-cpu-p2p")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    options = adapter.MooncakeBackendOptions(
        torch.zeros(world_size, dtype=torch.int32)
    )
    dist.init_process_group(backend="mooncake-python-cpu-p2p", pg_options=options)
    try:
        send = torch.tensor([rank], dtype=torch.int32)
        recv = torch.empty_like(send)
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send, (rank + 1) % world_size),
                dist.P2POp(dist.irecv, recv, (rank - 1) % world_size),
            ]
        )
        for work in works:
            work.wait()
        assert recv.item() == (rank - 1) % world_size
        print(f"rank {rank}: Mooncake Python ProcessGroup P2P smoke passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
