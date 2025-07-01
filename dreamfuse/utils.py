import os
import shutil
import threading
from pathlib import Path

import torch
from . import hdfs_utils as hdfs


def import_from_transformers_modules(
    pretrained_model_name_or_path, file_name, class_name
):
    import transformers

    module_path = transformers.dynamic_module_utils.get_cached_module_file(
        pretrained_model_name_or_path, file_name
    )
    return transformers.dynamic_module_utils.get_class_in_module(
        class_name, module_path
    )


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    import accelerate

    deepspeed_plugin = (
        accelerate.state.AcceleratorState().deepspeed_plugin
        if accelerate.state.is_initialized()
        else None
    )
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


def remove_excess_checkpoints(
    save_directory,
    checkpoints_total_limit: int = None,
    checkpoint_prefix="checkpoint",
    is_main_process: bool = True,
):
    # _after_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(save_directory)
        checkpoints = [d for d in checkpoints if d.startswith(checkpoint_prefix)]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[2]))

        # _after_ we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit` checkpoints
        if len(checkpoints) > checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit
            removing_checkpoints = checkpoints[0:num_to_remove]

            print(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            print(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_directory, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)


def sync_to_remote(
    local_root, remote_root, blocking=False, checkpoint_prefix="checkpoint"
):
    def _func(local_root, remote_root, checkpoint_prefix="checkpoint"):
        local_files = os.listdir(local_root)
        local_checkpoints = [_ for _ in local_files if _.startswith(checkpoint_prefix)]
        hdfs.mkdir(remote_root)
        remote_files = [Path(_).name for _ in hdfs.listdir(remote_root)]
        remote_checkpoints = [
            _ for _ in remote_files if _.startswith(checkpoint_prefix)
        ]
        for fn in list(set(local_files) - set(remote_checkpoints)):
            hdfs.copy(os.path.join(local_root, fn), remote_root)
        # remove remote file
        for fn in list(set(remote_checkpoints) - set(local_checkpoints)):
            hdfs.remove(os.path.join(remote_root, fn))
        print(f"sync {local_root} to {remote_root}")

    if blocking:
        _func(local_root, remote_root, checkpoint_prefix)
    else:
        threading.Thread(
            target=_func, args=(local_root, remote_root, checkpoint_prefix)
        ).start()


def is_distributed_training():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return True
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return world_size > 1


def contain_invalid_grad(optimizer):
    invalid_grad = False
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param.grad is not None:
                invalid_grad = invalid_grad or (
                    torch.isnan(param.grad).any()
                    or torch.isinf(param.grad).any()
                    or torch.isneginf(param.grad).any()
                )
    if is_distributed_training():
        invalid_grad_flag = torch.tensor(
            [1.0 if invalid_grad else 0.0],
            dtype=torch.float32,
            requires_grad=False,
        ).cuda()
        torch.distributed.all_reduce(
            invalid_grad_flag, op=torch.distributed.ReduceOp.MAX
        )
        invalid_grad = invalid_grad_flag.item() > 0
    return invalid_grad


def patch_npu_record_stream():
    torch.utils.rename_privateuse1_backend("npu")
    record_stream = torch.Tensor.record_stream

    def _func(*args, **kwargs):
        ret = record_stream(*args, **kwargs)
        torch.cuda.synchronize()
        return ret

    torch.Tensor.record_stream = _func


def patch_npu_diffusers_get_1d_rotary_pos_embed():
    from typing import Union
    import numpy as np
    import diffusers

    def __get_1d_rotary_pos_embed(
        dim: int,
        pos: Union[np.ndarray, int],
        theta: float = 10000.0,
        use_real=False,
        linear_factor=1.0,
        ntk_factor=1.0,
        repeat_interleave_real=True,
        freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
    ):
        assert dim % 2 == 0

        if isinstance(pos, int):
            pos = torch.arange(pos)
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)  # type: ignore  # [S]

        theta = theta * ntk_factor
        freqs = (
            1.0
            / (
                theta
                ** (
                    torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[
                        : (dim // 2)
                    ]
                    / dim
                )
            )
            / linear_factor
        )  # [D/2]
        freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
        if use_real and repeat_interleave_real:
            # flux, hunyuan-dit, cogvideox
            freqs_cos = (
                freqs.cos().float().repeat_interleave(2, dim=1).float()
            )  # [S, D]
            freqs_sin = (
                freqs.sin().float().repeat_interleave(2, dim=1).float()
            )  # [S, D]
            return freqs_cos, freqs_sin
        elif use_real:
            # stable audio
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
            return freqs_cos, freqs_sin
        else:
            # lumina
            freqs_cis = torch.polar(
                torch.ones_like(freqs), freqs
            )  # complex64     # [S, D/2]
            return freqs_cis

    diffusers.models.embeddings.get_1d_rotary_pos_embed = __get_1d_rotary_pos_embed
