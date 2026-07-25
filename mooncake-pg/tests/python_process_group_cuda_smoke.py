"""Two-rank CUDA Python ProcessGroup smoke backed by standalone pg_core."""

from __future__ import annotations

import importlib.util
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "mooncake_pg_python_probe", os.environ["MOONCAKE_PG_PYTHON_MODULE"]
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.register_cuda_backend("mooncake-python-cuda-smoke")

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    options = adapter.MooncakeBackendOptions(
        torch.zeros(world_size, dtype=torch.int32, device="cuda")
    )
    dist.init_process_group(backend="mooncake-python-cuda-smoke", pg_options=options)
    try:
        symmetric = symm_mem.empty(
            (8, 8), dtype=torch.bfloat16, device=f"cuda:{rank}"
        )
        symmetric_group = dist.group.WORLD.configure_symmetric_memory()
        symmetric_handle = symm_mem.rendezvous(symmetric, group=symmetric_group)
        assert symmetric_handle.rank == rank

        value = torch.tensor([rank + 1], dtype=torch.int32, device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        assert value.item() == 3

        value = torch.tensor([17 if rank == 0 else -1], dtype=torch.int32, device="cuda")
        dist.broadcast(value, src=0)
        torch.cuda.synchronize()
        assert value.item() == 17

        local = torch.tensor([rank], dtype=torch.int32, device="cuda")
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        torch.cuda.synchronize()
        assert [item.item() for item in gathered] == [0, 1]

        # Exceeds one 16 MiB staging slot across two ranks and exercises the
        # native core's CUDA all-gather chunking path.
        large_local = torch.full(
            ((1 << 22) + 17,), rank, dtype=torch.int32, device="cuda"
        )
        large_gathered = [torch.empty_like(large_local) for _ in range(world_size)]
        dist.all_gather(large_gathered, large_local)
        torch.cuda.synchronize()
        assert all(torch.all(item == peer) for peer, item in enumerate(large_gathered))

        rs_input = [torch.tensor([rank * world_size + peer], dtype=torch.int32, device="cuda") for peer in range(world_size)]
        rs_output = torch.empty_like(local)
        dist.reduce_scatter(rs_output, rs_input)
        torch.cuda.synchronize()
        assert rs_output.item() == 2 * rank + 2

        a2a_input = [torch.tensor([rank * 10 + peer], dtype=torch.int32, device="cuda") for peer in range(world_size)]
        a2a_output = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_to_all(a2a_output, a2a_input)
        torch.cuda.synchronize()
        assert [item.item() for item in a2a_output] == [rank, 10 + rank]

        reduced = torch.tensor([rank + 1], dtype=torch.int32, device="cuda")
        dist.reduce(reduced, dst=0)
        torch.cuda.synchronize()
        if rank == 0:
            assert reduced.item() == 3

        gathered_root = [torch.empty_like(local) for _ in range(world_size)] if rank == 0 else None
        dist.gather(local, gather_list=gathered_root, dst=0)
        torch.cuda.synchronize()
        if rank == 0:
            assert [item.item() for item in gathered_root] == [0, 1]

        scattered = torch.empty_like(local)
        scatter_list = [torch.tensor([8], dtype=torch.int32, device="cuda"), torch.tensor([9], dtype=torch.int32, device="cuda")] if rank == 0 else None
        dist.scatter(scattered, scatter_list=scatter_list, src=0)
        torch.cuda.synchronize()
        assert scattered.item() == 8 + rank

        send = torch.tensor([rank], dtype=torch.int32, device="cuda")
        recv = torch.empty_like(send)
        works = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send, (rank + 1) % world_size),
            dist.P2POp(dist.irecv, recv, (rank - 1) % world_size),
        ])
        for work in works:
            work.wait()
        torch.cuda.synchronize()
        assert recv.item() == (rank - 1) % world_size
        print(f"rank {rank}: native Python ProcessGroup CUDA smoke passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
