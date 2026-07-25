"""Two-rank CPU smoke for the experimental Python Mooncake ProcessGroup.

Run with ``torchrun --standalone --nproc-per-node=2`` and set
``MOONCAKE_PG_PYTHON_MODULE`` when loading the adapter from a source overlay.
"""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.distributed as dist


def load_adapter():
    adapter_path = os.getenv("MOONCAKE_PG_PYTHON_MODULE")
    if adapter_path is None:
        from mooncake import pg_python

        return pg_python

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
    backend_name = "mooncake-python-cpu-smoke"
    adapter.register_backend(backend_name)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    options = adapter.MooncakeBackendOptions(
        torch.zeros(world_size, dtype=torch.int32)
    )
    dist.init_process_group(backend=backend_name, pg_options=options)
    try:
        assert dist.get_world_size() == world_size
        tensor = torch.tensor([rank + 1], dtype=torch.int32)
        dist.all_reduce(tensor)
        assert tensor.item() == world_size * (world_size + 1) // 2
        dist.barrier()
        print(f"rank {rank}: Mooncake Python ProcessGroup CPU smoke passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
