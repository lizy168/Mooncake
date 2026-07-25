"""Two-rank CUDA P2P smoke for the Torch-free native Mooncake core."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

import pg_core


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    store = dist.TCPStore(
        "127.0.0.1",
        int(os.environ["MOONCAKE_CORE_STORE_PORT"]),
        world_size,
        rank == 0,
        timeout=timedelta(seconds=60),
    )
    core = pg_core.P2PCore(rank, world_size, store, "127.0.0.1", rank)
    try:
        send = torch.tensor([rank], dtype=torch.int32, device="cuda")
        recv = torch.empty_like(send)
        stream = torch.cuda.current_stream().cuda_stream
        core.allreduce_cuda(
            send.data_ptr(), recv.data_ptr(), send.nbytes,
            3, 0, stream,
        )
        torch.cuda.synchronize()
        assert recv.item() == sum(range(world_size))
        print(f"rank {rank}: native Torch-free CUDA all-reduce smoke passed")
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
