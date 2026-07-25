"""Two-rank direct P2P smoke for the Torch-free native Mooncake core.

Torch is deliberately used only here, at the Python boundary, for TCPStore
and for allocating CPU buffers.  ``pg_core`` itself receives just raw buffer
addresses and byte counts and has no Torch dynamic dependency.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

import pg_core


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    store = dist.TCPStore(
        "127.0.0.1",
        int(os.environ["MOONCAKE_CORE_STORE_PORT"]),
        world_size,
        rank == 0,
        timeout=timedelta(seconds=60),
    )
    core = pg_core.P2PCore(rank, world_size, store, "127.0.0.1")
    try:
        send = torch.tensor([rank], dtype=torch.int32)
        recv = torch.empty_like(send)
        peer = (rank + 1) % world_size
        works = [
            core.recv(recv.data_ptr(), recv.nbytes, (rank - 1) % world_size),
            core.send(send.data_ptr(), send.nbytes, peer),
        ]
        for work in works:
            assert work.wait(30_000)
            assert work.is_success()
        assert recv.item() == (rank - 1) % world_size
        print(f"rank {rank}: native Torch-free P2P core smoke passed")
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
