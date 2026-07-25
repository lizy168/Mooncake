"""Experimental Python ``ProcessGroup`` adapter for Mooncake PG.

This module is a feasibility probe for PyTorch's ``PyProcessGroup``
trampoline.  It deliberately keeps the current native Mooncake ProcessGroup
as its delegated core; extracting that core from Torch is a separate step.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from mooncake import pg as _native_pg

MooncakeBackendOptions = _native_pg.MooncakeBackendOptions


class _CoreWork(dist.Work):
    """PyTorch Work adapter that retains buffers for a native core call."""

    def __init__(
        self,
        core_work: Any,
        tensors: list[torch.Tensor],
        *,
        cuda: bool = False,
        on_wait: Any | None = None,
    ) -> None:
        super().__init__()
        self._core_work = core_work
        self._tensors = tensors
        self._cuda = cuda
        self._on_wait = on_wait

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        timeout_ms = int(timeout.total_seconds() * 1_000)
        if self._cuda:
            completed = self._core_work.wait(
                timeout_ms, torch.cuda.current_stream().cuda_stream
            )
        else:
            completed = self._core_work.wait(timeout_ms)
        if completed and self._on_wait is not None:
            # CUDA Work.wait() orders the caller's current stream after the
            # core completion event. Copying here keeps reshaped list-style
            # collective outputs ordered after the native transfer.
            self._on_wait()
        return completed


class _ImmediateCoreWork(dist.Work):
    def __init__(self, tensors: list[torch.Tensor]) -> None:
        super().__init__()
        self._tensors = tensors

    def wait(self, timeout: timedelta = timedelta(0)) -> bool:
        del timeout
        return True


def _core_dtype(tensor: torch.Tensor) -> int:
    mapping = {
        torch.uint8: 0,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.float32: 5,
        torch.float64: 6,
        torch.bool: 7,
        torch.bfloat16: 8,
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as exc:
        raise RuntimeError(f"unsupported Mooncake core dtype: {tensor.dtype}") from exc


def _core_reduce_op(op: Any) -> int:
    op = getattr(op, "op", op)
    mapping = {
        dist.ReduceOp.RedOpType.SUM: 0,
        dist.ReduceOp.RedOpType.PRODUCT: 1,
        dist.ReduceOp.RedOpType.MIN: 2,
        dist.ReduceOp.RedOpType.MAX: 3,
    }
    try:
        return mapping[op]
    except KeyError as exc:
        raise RuntimeError(f"unsupported Mooncake core reduce op: {op}") from exc


class MooncakePythonProcessGroup(dist.ProcessGroup):
    """Torch trampoline adapter backed directly by the native ``pg_core``."""

    _core_module: Any | None = None

    @classmethod
    def _load_core(cls) -> Any:
        if cls._core_module is None:
            try:
                import pg_core
            except ImportError as exc:
                raise ImportError("Mooncake Python ProcessGroup requires pg_core") from exc
            cls._core_module = pg_core
        return cls._core_module

    @classmethod
    def set_transfer_engine(cls, engine: Any | None) -> None:
        core = cls._load_core()
        core.set_transfer_engine(0 if engine is None else engine.get_engine_ptr())

    @classmethod
    def set_host_ip(cls, host_ip: str) -> None:
        cls._load_core().set_host_ip(host_ip)

    @classmethod
    def set_device_filter(cls, filters: list[str]) -> None:
        cls._load_core().set_device_filter(filters)

    @classmethod
    def get_preferred_hca(cls, location: str) -> str:
        return cls._load_core().get_preferred_hca(location)

    def __init__(
        self, dist_backend_opts: Any, backend_options: Any, *, cuda: bool = False
    ) -> None:
        # Torch 2.11's Store-taking Python binding does not construct the
        # PyProcessGroup trampoline for subclasses. The native core receives
        # the Store directly, so use the alias-capable rank/size constructor.
        super().__init__(dist_backend_opts.group_rank, dist_backend_opts.group_size)
        del backend_options
        self._rank = dist_backend_opts.group_rank
        self._size = dist_backend_opts.group_size
        self._cuda = cuda
        self._cuda_device = torch.cuda.current_device() if cuda else -1
        # ``group_id`` is PyTorch's assigned ProcessGroup name for extended
        # backend creators. Keep it on the Python backend as well: SGLang's
        # symmetric-memory rendezvous consumes this public group identity.
        self._group_name = str(dist_backend_opts.group_id)
        self._set_group_name(self._group_name)
        try:
            self._backend_index = int(self._group_name)
        except ValueError as exc:
            raise RuntimeError(
                f"Mooncake PG requires a numeric PyTorch group id, got "
                f"{self._group_name!r}"
            ) from exc
        # Native ProcessGroup::setGroupName registers this group with Torch's
        # symmetric-memory allocator. PyProcessGroup dispatches that virtual
        # call to ``setGroupName`` above, so mirror the registration here with
        # the rendezvous Store supplied by PyTorch's extended backend API.
        global_ranks = list(dist_backend_opts.global_ranks_in_group)
        if not global_ranks:
            global_ranks = list(range(self._size))
        rank_key = "_".join(str(rank) for rank in sorted(global_ranks))
        self._symm_mem_store = dist.PrefixStore(
            f"symmetric_memory-{rank_key}", dist_backend_opts.store
        )
        core = self._load_core()
        self._core = core.P2PCore(
            self._rank, self._size, dist_backend_opts.store, "127.0.0.1",
            self._cuda_device, self._backend_index,
        )

    def getBackendName(self) -> str:
        # This must match the registered public backend name. PyTorch uses it
        # when resolving the ProcessGroup backend for a tensor device.
        return "mooncake" if self._cuda else "mooncake-cpu"

    def setGroupName(self, name: str) -> None:
        # PyProcessGroup dispatches ProcessGroup._set_group_name() back into
        # this override instead of retaining a C++ base-class field.
        self._group_name = name

    def getGroupName(self) -> str:
        return self._group_name

    def symmetric_memory_group_name(self) -> str:
        # PyTorch first registers every ProcessGroup name with its native
        # symmetric-memory registry. A PyProcessGroup has no native Store, so
        # use a separate logical name for the adapter-owned Store instead.
        return f"mooncake-python-symm-{self._group_name}"

    def configure_symmetric_memory(self) -> str:
        name = self.symmetric_memory_group_name()
        if not hasattr(self, "_symm_mem_pg"):
            # Rendezvous resolves the logical name through Torch's native
            # registry. This metadata-only ProcessGroup carries the Store;
            # Mooncake collectives continue to use the Python adapter/core.
            self._symm_mem_pg = dist.ProcessGroup(
                self._symm_mem_store, self._rank, self._size
            )
            self._symm_mem_pg._set_group_name(name)
            torch._C._distributed_c10d._register_process_group(
                name, self._symm_mem_pg
            )
        torch._C._distributed_c10d._SymmetricMemory.set_group_info(
            name, self._rank, self._size, self._symm_mem_store
        )
        return name

    def getRank(self) -> int:
        return self._rank

    def getSize(self) -> int:
        # PyProcessGroup's C++ trampoline looks up this exact method name.
        return self._size

    def _one(self, tensors: list[Any], what: str) -> torch.Tensor:
        if len(tensors) != 1:
            raise RuntimeError(f"{what} requires one tensor")
        tensor = tensors[0]
        expected_device = "cuda" if self._cuda else "cpu"
        if tensor.device.type != expected_device:
            raise RuntimeError(f"{what} requires {expected_device} tensors")
        return tensor

    def _collective(
        self,
        op: int,
        input_tensor: torch.Tensor | None,
        output_tensor: torch.Tensor | None,
        unit_bytes: int,
        opts: Any | None = None,
        root_rank: int = 0,
        keepalive: list[torch.Tensor] | None = None,
        on_complete: Any | None = None,
    ) -> dist.Work:
        input_contiguous = (
            None if input_tensor is None else input_tensor.contiguous()
        )
        output_contiguous = (
            None if output_tensor is None else output_tensor.contiguous()
        )
        dtype_source = (
            input_contiguous if input_contiguous is not None else output_contiguous
        )
        assert dtype_source is not None or op == 8
        args = (
            op,
            0 if input_contiguous is None else input_contiguous.data_ptr(),
            0 if input_contiguous is None else input_contiguous.nbytes,
            0 if output_contiguous is None else output_contiguous.data_ptr(),
            0 if output_contiguous is None else output_contiguous.nbytes,
            unit_bytes,
            0 if dtype_source is None else _core_dtype(dtype_source),
            0 if opts is None else _core_reduce_op(opts.reduceOp),
            root_rank,
        )
        if self._cuda:
            assert dtype_source is not None
            core_work = self._core.collective_cuda(
                *args, torch.cuda.current_stream(dtype_source.device).cuda_stream
            )
        else:
            self._core.collective_cpu(*args)
            core_work = None
        def copy_back() -> None:
            if output_contiguous is not None and output_tensor is not None and (
                output_contiguous.data_ptr() != output_tensor.data_ptr()
            ):
                output_tensor.copy_(output_contiguous)
            if on_complete is not None:
                on_complete()

        tensors = [tensor for tensor in (input_contiguous, output_contiguous) if tensor is not None]
        if keepalive:
            tensors.extend(keepalive)
        if core_work is not None:
            return _CoreWork(core_work, tensors, cuda=True, on_wait=copy_back)
        copy_back()
        return _ImmediateCoreWork(tensors)

    def allreduce(self, tensors: list[Any], opts: Any) -> dist.Work:
        if len(tensors) != 1:
            raise RuntimeError("native core allreduce requires one tensor")
        tensor = tensors[0].contiguous()
        if self._cuda:
            if tensor.device.type != "cuda":
                raise RuntimeError("CUDA Mooncake ProcessGroup requires CUDA tensors")
            core_work = self._core.allreduce_cuda(
                tensor.data_ptr(), tensor.data_ptr(), tensor.nbytes,
                _core_dtype(tensor), _core_reduce_op(opts.reduceOp),
                torch.cuda.current_stream(tensor.device).cuda_stream,
            )
        else:
            if tensor.device.type != "cpu":
                raise RuntimeError("CPU Mooncake ProcessGroup requires CPU tensors")
            self._core.allreduce_cpu(
                tensor.data_ptr(), tensor.data_ptr(), tensor.nbytes,
                _core_dtype(tensor), _core_reduce_op(opts.reduceOp)
            )
        def copy_back() -> None:
            if tensor.data_ptr() != tensors[0].data_ptr():
                tensors[0].copy_(tensor)
        if self._cuda:
            return _CoreWork(core_work, [tensor], cuda=True, on_wait=copy_back)
        copy_back()
        return _ImmediateCoreWork([tensor])

    def broadcast(self, tensors: list[Any], opts: Any) -> dist.Work:
        tensor = self._one(tensors, "broadcast")
        root = opts.rootRank
        return self._collective(0, tensor if self._rank == root else None, tensor,
                                tensor.nbytes, root_rank=root)

    def allgather(
        self, output_tensors: list[list[Any]], input_tensors: list[Any], opts: Any
    ) -> dist.Work:
        input_tensor = self._one(input_tensors, "allgather")
        if len(output_tensors) != 1 or len(output_tensors[0]) != self._size:
            raise RuntimeError("allgather requires one output tensor per rank")
        flat = torch.empty(
            self._size * input_tensor.numel(), dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        def unpack() -> None:
            for index, tensor in enumerate(output_tensors[0]):
                tensor.copy_(
                    flat.narrow(
                        0, index * input_tensor.numel(), input_tensor.numel()
                    ).view_as(tensor)
                )

        return self._collective(
            2, input_tensor, flat, input_tensor.nbytes, on_complete=unpack
        )

    def reduce_scatter(
        self, output_tensors: list[Any], input_tensors: list[list[Any]], opts: Any
    ) -> dist.Work:
        output = self._one(output_tensors, "reduce_scatter")
        if len(input_tensors) != 1 or len(input_tensors[0]) != self._size:
            raise RuntimeError("reduce_scatter requires one input tensor per rank")
        inputs = input_tensors[0]
        if any(tensor.numel() != output.numel() or tensor.dtype != output.dtype for tensor in inputs):
            raise RuntimeError("reduce_scatter requires input tensors matching output")
        flat = torch.cat([tensor.contiguous().view(-1) for tensor in inputs])
        return self._collective(3, flat, output, output.nbytes, opts)

    def barrier(self, opts: Any) -> dist.Work:
        del opts
        if self._cuda:
            return _CoreWork(
                self._core.collective_cuda(
                    8, stream=torch.cuda.current_stream().cuda_stream
                ),
                [],
                cuda=True,
            )
        else:
            self._core.collective_cpu(8)
        return _ImmediateCoreWork([])

    def reduce(self, tensors: list[Any], opts: Any) -> dist.Work:
        tensor = self._one(tensors, "reduce")
        root = opts.rootRank
        return self._collective(5, tensor, tensor if self._rank == root else None,
                                tensor.nbytes, opts, root)

    def gather(
        self, output_tensors: list[list[Any]], input_tensors: list[Any], opts: Any
    ) -> dist.Work:
        input_tensor = self._one(input_tensors, "gather")
        root = opts.rootRank
        flat = None
        outputs: list[torch.Tensor] = []
        if self._rank == root:
            if len(output_tensors) != 1 or len(output_tensors[0]) != self._size:
                raise RuntimeError("gather root requires one output tensor per rank")
            outputs = output_tensors[0]
            flat = torch.empty(
                self._size * input_tensor.numel(), dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
        def unpack() -> None:
            if flat is not None:
                for index, tensor in enumerate(outputs):
                    tensor.copy_(
                        flat.narrow(
                            0, index * input_tensor.numel(), input_tensor.numel()
                        ).view_as(tensor)
                    )

        return self._collective(
            6,
            input_tensor,
            flat,
            input_tensor.nbytes,
            root_rank=root,
            on_complete=unpack,
        )

    def scatter(
        self, output_tensors: list[Any], input_tensors: list[list[Any]], opts: Any
    ) -> dist.Work:
        output = self._one(output_tensors, "scatter")
        root = opts.rootRank
        flat = None
        if self._rank == root:
            if len(input_tensors) != 1 or len(input_tensors[0]) != self._size:
                raise RuntimeError("scatter root requires one input tensor per rank")
            flat = torch.cat([tensor.contiguous().view(-1) for tensor in input_tensors[0]])
        return self._collective(7, flat, output, output.nbytes, root_rank=root)

    def alltoall(
        self, output_tensors: list[Any], input_tensors: list[Any], opts: Any
    ) -> dist.Work:
        if len(input_tensors) != self._size or len(output_tensors) != self._size:
            raise RuntimeError("alltoall requires one input/output tensor per rank")
        unit = input_tensors[0]
        if any(tensor.numel() != unit.numel() or tensor.dtype != unit.dtype for tensor in input_tensors):
            raise RuntimeError("alltoall requires equal input tensor shapes and dtypes")
        flat_input = torch.cat([tensor.contiguous().view(-1) for tensor in input_tensors])
        flat_output = torch.empty_like(flat_input)
        def unpack() -> None:
            for index, tensor in enumerate(output_tensors):
                tensor.copy_(
                    flat_output.narrow(
                        0, index * unit.numel(), unit.numel()
                    ).view_as(tensor)
                )

        return self._collective(
            4, flat_input, flat_output, unit.nbytes, on_complete=unpack
        )

    def send(self, tensors: list[Any], dst_rank: int, tag: int) -> dist.Work:
        if self._core is not None:
            del tag
            if len(tensors) != 1:
                raise RuntimeError("Mooncake P2P supports one tensor per operation")
            tensor = tensors[0]
            if self._cuda != (tensor.device.type == "cuda"):
                raise RuntimeError("tensor device does not match Mooncake ProcessGroup")
            contiguous = tensor.contiguous()
            return _CoreWork(
                self._core.send(
                    contiguous.data_ptr(), contiguous.nbytes, dst_rank,
                    0 if not self._cuda else torch.cuda.current_stream(contiguous.device).cuda_stream,
                ),
                [contiguous],
            )

    def recv(self, tensors: list[Any], src_rank: int, tag: int) -> dist.Work:
        if self._core is not None:
            del tag
            if len(tensors) != 1:
                raise RuntimeError("Mooncake P2P supports one tensor per operation")
            tensor = tensors[0]
            if self._cuda != (tensor.device.type == "cuda") or not tensor.is_contiguous():
                raise RuntimeError("tensor device or layout does not match Mooncake ProcessGroup")
            return _CoreWork(
                self._core.recv(
                    tensor.data_ptr(), tensor.nbytes, src_rank,
                    0 if not self._cuda else torch.cuda.current_stream(tensor.device).cuda_stream,
                ),
                [tensor],
            )

    @property
    def native_backend(self) -> dist.ProcessGroup:
        """Return the native delegated backend for probe-only helper calls."""
        return self._native


def register_backend(name: str = "mooncake-python-cpu") -> None:
    """Register the experimental CPU backend with PyTorch.

    Registration is explicit so importing the module does not alter the
    process-wide backend registry.
    """
    dist.Backend.register_backend(
        name,
        MooncakePythonProcessGroup,
        extended_api=True,
        devices=["cpu"],
    )


class MooncakePythonCudaProcessGroup(MooncakePythonProcessGroup):
    def __init__(self, dist_backend_opts: Any, backend_options: Any) -> None:
        super().__init__(dist_backend_opts, backend_options, cuda=True)


def register_cuda_backend(name: str = "mooncake-python-cuda") -> None:
    dist.Backend.register_backend(
        name,
        MooncakePythonCudaProcessGroup,
        extended_api=True,
        devices=["cuda"],
    )


def _native_backend(group: dist.ProcessGroup | None) -> dist.ProcessGroup:
    resolved = dist.group.WORLD if group is None else group
    if not isinstance(resolved, MooncakePythonProcessGroup):
        raise TypeError("group must be a MooncakePythonProcessGroup")
    return resolved.native_backend


def get_active_ranks(group: dist.ProcessGroup | None = None) -> Any:
    return _native_pg.get_active_ranks(_native_backend(group))


def extend_group_size_to(group: dist.ProcessGroup | None, size: int) -> None:
    _native_pg.extend_group_size_to(_native_backend(group), size)
