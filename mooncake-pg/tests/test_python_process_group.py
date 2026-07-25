"""CPU feasibility tests for the Python ProcessGroup adapter.

The first test isolates PyTorch's trampoline contract.  The second uses the
current Mooncake CPU ProcessGroup as a delegated native core.  It intentionally
does not cover P2P: Python ProcessGroup subclasses cannot populate the C++
backend map used by PyTorch's P2P dispatcher.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import AllreduceOptions, BarrierOptions


class _ImmediateWork(dist.Work):
    def __init__(self) -> None:
        super().__init__()

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        return True


class _TrampolineProbeProcessGroup(dist.ProcessGroup):
    def __init__(self, rank: int, size: int) -> None:
        super().__init__(rank, size)
        self._size = size

    def getSize(self) -> int:
        return self._size

    def getBackendName(self) -> str:
        return "trampoline-probe"

    def allreduce(self, tensors: list[torch.Tensor], opts: AllreduceOptions) -> dist.Work:
        return _ImmediateWork()

    def barrier(self, opts: BarrierOptions) -> dist.Work:
        return _ImmediateWork()


def _create_trampoline_probe(
    store: dist.Store, rank: int, world_size: int, timeout: timedelta
) -> _TrampolineProbeProcessGroup:
    del store, timeout
    return _TrampolineProbeProcessGroup(rank, world_size)


class TestPythonProcessGroupTrampoline(unittest.TestCase):
    def test_get_size_override_and_python_work(self) -> None:
        backend_name = "trampoline-probe"
        dist.Backend.register_backend(
            backend_name, _create_trampoline_probe, devices=["cpu"]
        )
        dist.init_process_group(
            backend=backend_name,
            init_method="tcp://127.0.0.1:29590",
            rank=0,
            world_size=1,
        )
        try:
            group = dist.group.WORLD
            self.assertIsInstance(group, _TrampolineProbeProcessGroup)
            self.assertEqual(group.size(), 1)

            # This enters the C++ ProcessGroup virtual call and returns through
            # PyProcessGroup's C++ PyWorkHolder, not a direct Python call.
            tensor = torch.tensor([3])
            work = dist.all_reduce(tensor, async_op=True)
            self.assertIsInstance(work, dist.Work)
            self.assertTrue(work.wait())
            dist.barrier()

            # ``size()`` enters C++, then virtual-dispatches to ``getSize()``.
            group._size = 5
            self.assertEqual(dist.get_world_size(), 5)
        finally:
            dist.destroy_process_group()


class TestMooncakePythonProcessGroup(unittest.TestCase):
    def test_cpu_collectives_delegate_to_native_backend(self) -> None:
        try:
            adapter_path = os.getenv("MOONCAKE_PG_PYTHON_MODULE")
            if adapter_path is None:
                from mooncake import pg_python
            else:
                module_spec = importlib.util.spec_from_file_location(
                    "mooncake_pg_python_probe", adapter_path
                )
                if module_spec is None or module_spec.loader is None:
                    raise ImportError(f"cannot load adapter from {adapter_path}")
                pg_python = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(pg_python)
        except ImportError as exc:  # pragma: no cover - test environment issue
            self.skipTest(f"Mooncake PG extension is unavailable: {exc}")

        backend_name = "mooncake-python-cpu-probe"
        pg_python.register_backend(backend_name)
        active_ranks = torch.zeros(1, dtype=torch.int32)
        options = pg_python.MooncakeBackendOptions(active_ranks)

        dist.init_process_group(
            backend=backend_name,
            init_method="tcp://127.0.0.1:29591",
            rank=0,
            world_size=1,
            pg_options=options,
        )
        try:
            self.assertIsInstance(dist.group.WORLD, pg_python.MooncakePythonProcessGroup)
            self.assertEqual(dist.get_world_size(), 1)
            self.assertEqual(dist.group.WORLD.group_name, "0")

            tensor = torch.tensor([7], dtype=torch.int32)
            dist.all_reduce(tensor)
            self.assertEqual(tensor.item(), 7)
            dist.barrier()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
