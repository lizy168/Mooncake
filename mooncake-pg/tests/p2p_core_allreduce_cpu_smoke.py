"""Two-rank raw CPU all-reduce smoke for the Torch-free native PG core."""

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
        input_tensor = torch.tensor([rank + 1, 2], dtype=torch.int32)
        output_tensor = torch.empty_like(input_tensor)
        # ScalarType::kInt32 and ReduceOp::kProduct are core enum values.
        core.allreduce_cpu(
            input_tensor.data_ptr(), output_tensor.data_ptr(), input_tensor.nbytes,
            3, 1,
        )
        assert output_tensor.tolist() == [2, 4]
        print(f"rank {rank}: native Torch-free CPU all-reduce smoke passed")
    finally:
        core.shutdown()


if __name__ == "__main__":
    main()
