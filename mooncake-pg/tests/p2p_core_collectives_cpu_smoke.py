"""Raw CPU collective coverage for the Torch-free native PG core."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

import pg_core

DTYPE_INT32 = 3
SUM = 0
BROADCAST, ALLREDUCE, ALLGATHER, REDUCE_SCATTER, ALLTOALL, REDUCE, GATHER, SCATTER, BARRIER = range(9)


def call(core, op, input_tensor=None, output_tensor=None, unit_bytes=4, root=0):
    core.collective_cpu(
        op,
        0 if input_tensor is None else input_tensor.data_ptr(),
        0 if input_tensor is None else input_tensor.nbytes,
        0 if output_tensor is None else output_tensor.data_ptr(),
        0 if output_tensor is None else output_tensor.nbytes,
        unit_bytes,
        DTYPE_INT32,
        SUM,
        root,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    host_ip = os.environ.get("MOONCAKE_CORE_HOST_IP", "127.0.0.1")
    device_filters = os.environ.get("MOONCAKE_PGTEST_DEVICE_FILTERS", "")
    pg_core.set_host_ip(host_ip)
    if device_filters:
        pg_core.set_device_filter(
            [item for item in device_filters.split(",") if item]
        )
    store = dist.TCPStore("127.0.0.1", int(os.environ["MOONCAKE_CORE_STORE_PORT"]), world_size, rank == 0, timeout=timedelta(seconds=60))
    core = pg_core.P2PCore(rank, world_size, store, host_ip)
    try:
        broadcast_in = torch.tensor([11], dtype=torch.int32) if rank == 0 else None
        broadcast_out = torch.empty(1, dtype=torch.int32)
        call(core, BROADCAST, broadcast_in, broadcast_out)
        assert broadcast_out.tolist() == [11]

        gathered_in = torch.tensor([rank], dtype=torch.int32)
        gathered_out = torch.empty(world_size, dtype=torch.int32)
        call(core, ALLGATHER, gathered_in, gathered_out)
        assert gathered_out.tolist() == [0, 1]

        rs_in = torch.tensor([rank * 2, rank * 2 + 1], dtype=torch.int32)
        rs_out = torch.empty(1, dtype=torch.int32)
        call(core, REDUCE_SCATTER, rs_in, rs_out)
        assert rs_out.tolist() == [2 * rank + 2]

        a2a_in = torch.tensor([rank * 10, rank * 10 + 1], dtype=torch.int32)
        a2a_out = torch.empty_like(a2a_in)
        call(core, ALLTOALL, a2a_in, a2a_out)
        assert a2a_out.tolist() == [rank, 10 + rank]

        reduce_in = torch.tensor([rank + 1], dtype=torch.int32)
        reduce_out = torch.empty_like(reduce_in) if rank == 0 else None
        call(core, REDUCE, reduce_in, reduce_out)
        if rank == 0:
            assert reduce_out.tolist() == [3]

        gather_in = torch.tensor([rank], dtype=torch.int32)
        gather_out = torch.empty(world_size, dtype=torch.int32) if rank == 0 else None
        call(core, GATHER, gather_in, gather_out)
        if rank == 0:
            assert gather_out.tolist() == [0, 1]

        scatter_in = torch.tensor([7, 8], dtype=torch.int32) if rank == 0 else None
        scatter_out = torch.empty(1, dtype=torch.int32)
        call(core, SCATTER, scatter_in, scatter_out)
        assert scatter_out.tolist() == [7 + rank]

        call(core, BARRIER, None, None, 0)
        print(f"rank {rank}: native Torch-free CPU collective smoke passed")
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
