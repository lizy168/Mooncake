"""Opt-in SGLang startup hook for the Torch-free PG prototype."""

import os
import importlib.util
import sys

import torch
import torch.distributed as dist

if os.environ.get("MOONCAKE_PG_PYTHON_PROTOTYPE") == "1":
    module_path = os.environ.get("MOONCAKE_PG_PYTHON_MODULE")
    if module_path:
        spec = importlib.util.spec_from_file_location("mooncake.pg_python", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Mooncake PG adapter from {module_path}")
        pg_python = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = pg_python
        spec.loader.exec_module(pg_python)
    else:
        from mooncake import pg_python

    pg_python.register_cuda_backend("mooncake")
    pg_python.register_backend("mooncake-cpu")

    # Keep the Python adapter on the same configuration and ownership path as
    # the public Mooncake PG API. The native core receives only a raw engine
    # handle; it never imports or links Torch.
    from mooncake import ep as mooncake_ep

    _set_host_ip = mooncake_ep.set_host_ip
    _set_device_filter = mooncake_ep.set_device_filter
    _set_transfer_engine = mooncake_ep.set_transfer_engine

    def set_host_ip(host_ip):
        _set_host_ip(host_ip)
        pg_python.MooncakePythonProcessGroup.set_host_ip(host_ip)

    def set_device_filter(filters):
        _set_device_filter(filters)
        pg_python.MooncakePythonProcessGroup.set_device_filter(filters)

    def set_transfer_engine(engine):
        _set_transfer_engine(engine)
        pg_python.MooncakePythonProcessGroup.set_transfer_engine(engine)

    mooncake_ep.set_host_ip = set_host_ip
    mooncake_ep.set_device_filter = set_device_filter
    mooncake_ep.set_transfer_engine = set_transfer_engine

    from mooncake import pg as mooncake_pg

    _get_preferred_hca = mooncake_pg.get_preferred_hca

    def get_preferred_hca(group, location):
        if isinstance(group, pg_python.MooncakePythonProcessGroup):
            return pg_python.MooncakePythonProcessGroup.get_preferred_hca(
                location
            )
        return _get_preferred_hca(group, location)

    mooncake_pg.get_preferred_hca = get_preferred_hca

    # Torch 2.11 routes the tensor forms through non-overridable C++ base
    # methods. Lower these public APIs to the list forms for Python PGs; the
    # latter correctly trampoline to Python collective overrides. Doing this
    # at the public API boundary covers both GroupCoordinator and the direct
    # DP-attention CPU-group call sites without changing SGLang source.
    _all_gather_into_tensor = dist.all_gather_into_tensor
    _reduce_scatter_tensor = dist.reduce_scatter_tensor

    def _is_python_group(group):
        resolved = dist.group.WORLD if group is None else group
        return isinstance(resolved, pg_python.MooncakePythonProcessGroup)

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        if not _is_python_group(group):
            return _all_gather_into_tensor(output, input, group, async_op)
        resolved = dist.group.WORLD if group is None else group
        return dist.all_gather(
            list(output.chunk(resolved.size())),
            input,
            group=group,
            async_op=async_op,
        )

    def reduce_scatter_tensor(
        output, input, op=dist.ReduceOp.SUM, group=None, async_op=False
    ):
        if not _is_python_group(group):
            return _reduce_scatter_tensor(output, input, op, group, async_op)
        resolved = dist.group.WORLD if group is None else group
        return dist.reduce_scatter(
            output,
            list(input.chunk(resolved.size())),
            op=op,
            group=group,
            async_op=async_op,
        )

    dist.all_gather_into_tensor = all_gather_into_tensor
    dist.reduce_scatter_tensor = reduce_scatter_tensor

    # SGLang's NVLink symmetric-memory all-gather works through the public
    # ProcessGroup object. Its exact-type assertion unnecessarily rejects the
    # PyProcessGroup trampoline despite the required rank/size/rendezvous API
    # being present. Keep the implementation identical apart from accepting a
    # ProcessGroup subclass, so the prototype retains this decode fast path.
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    def create_multimem_state(
        group, rank_in_group, max_tokens, hidden_size, device=None
    ):
        assert isinstance(group, dist.ProcessGroup), (
            f"Expected ProcessGroup, got {type(group)}"
        )
        assert hidden_size % triton_symm_mem_ag._NUMEL_PER_THREAD == 0, (
            f"hidden_size={hidden_size} must be a multiple of "
            f"{triton_symm_mem_ag._NUMEL_PER_THREAD} bf16 for 16-byte "
            "multimem.st row alignment"
        )
        device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        pad_bytes = triton_symm_mem_ag._MAX_BLOCKS * group.size() * 4
        symm_mem = triton_symm_mem_ag.symm_mem
        symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
        with torch.inference_mode(False), torch.no_grad():
            comm_buff = symm_mem.empty(
                (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
            )
        if isinstance(group, pg_python.MooncakePythonProcessGroup):
            symm_group = group.configure_symmetric_memory()
            handle = symm_mem.rendezvous(comm_buff, group=symm_group)
        else:
            handle = symm_mem.rendezvous(comm_buff, group=group)
        assert handle.rank == rank_in_group, (
            f"symm_mem handle rank {handle.rank} != rank_in_group {rank_in_group}"
        )
        return triton_symm_mem_ag.MultimemAllGatherState(
            group=group,
            rank_in_group=rank_in_group,
            world_size=group.size(),
            device=device,
            max_token_num=max_tokens,
            hidden_dim=hidden_size,
            comm_buff=comm_buff,
            symm_mem_hdl=handle,
        )

    triton_symm_mem_ag.create_state = create_multimem_state

    # The installed EP helper assumes its argument is the legacy C++ backend.
    # This healthy E2E path has no failure/rejoin event, so every rank is active.
    def get_active_ranks(group):
        return torch.ones(group.size(), dtype=torch.bool)

    mooncake_ep.get_active_ranks = get_active_ranks
