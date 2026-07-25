"""Two-rank ProcessGroup-level CPU collective coverage for ``pg_core``."""

from __future__ import annotations

import importlib
import os

import torch
import torch.distributed as dist


def load_adapter():
    # sitecustomize loads the same canonical module used by the SGLang probe.
    # Reusing it preserves the class identity used by the public tensor-API
    # compatibility hook.
    return importlib.import_module("mooncake.pg_python")


def main() -> None:
    adapter = load_adapter()
    adapter.register_backend("mooncake-python-core-collectives")
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    options = adapter.MooncakeBackendOptions(torch.zeros(world_size, dtype=torch.int32))
    dist.init_process_group(backend="mooncake-python-core-collectives", pg_options=options)
    try:
        value = torch.tensor([17 if rank == 0 else -1], dtype=torch.int32)
        dist.broadcast(value, src=0)
        assert value.tolist() == [17]

        local = torch.tensor([rank], dtype=torch.int32)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        assert [item.item() for item in gathered] == [0, 1]

        gathered_tensor = torch.empty(world_size, dtype=torch.int32)
        dist.all_gather_into_tensor(gathered_tensor, local)
        assert gathered_tensor.tolist() == [0, 1]

        rs_input = [torch.tensor([rank * world_size + peer], dtype=torch.int32) for peer in range(world_size)]
        rs_output = torch.empty_like(local)
        dist.reduce_scatter(rs_output, rs_input)
        assert rs_output.tolist() == [2 * rank + 2]

        rs_tensor_input = torch.tensor(
            [rank * world_size + peer for peer in range(world_size)],
            dtype=torch.int32,
        )
        dist.reduce_scatter_tensor(rs_output, rs_tensor_input)
        assert rs_output.tolist() == [2 * rank + 2]

        reduced = torch.tensor([rank + 1], dtype=torch.int32)
        dist.reduce(reduced, dst=0)
        if rank == 0:
            assert reduced.tolist() == [3]

        gathered_root = [torch.empty_like(local) for _ in range(world_size)] if rank == 0 else None
        dist.gather(local, gather_list=gathered_root, dst=0)
        if rank == 0:
            assert [item.item() for item in gathered_root] == [0, 1]

        scattered = torch.empty_like(local)
        scatter_list = [torch.tensor([8], dtype=torch.int32), torch.tensor([9], dtype=torch.int32)] if rank == 0 else None
        dist.scatter(scattered, scatter_list=scatter_list, src=0)
        assert scattered.tolist() == [8 + rank]

        a2a_input = [torch.tensor([rank * 10 + peer], dtype=torch.int32) for peer in range(world_size)]
        a2a_output = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_to_all(a2a_output, a2a_input)
        assert [item.item() for item in a2a_output] == [rank, 10 + rank]
        print(f"rank {rank}: native Python ProcessGroup CPU collectives passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
