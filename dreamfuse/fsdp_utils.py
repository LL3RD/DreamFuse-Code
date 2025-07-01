import os
import re
import json
import gc
import functools
import contextlib
from typing import Dict, Union, Optional, Type, Set

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType,
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
import torch.distributed.checkpoint as torch_dcp
import torch.distributed.checkpoint.state_dict
from torch.distributed.fsdp.api import (
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
import accelerate
import safetensors
import diffusers
import transformers
from huggingface_hub.serialization import split_torch_state_dict_into_shards

from .ema_utils import EMAModel


def upcast_trainable_param_to_fp32_(fsdp_model):
    for m in FSDP.fsdp_modules(fsdp_model):
        if m._has_params:
            param = m._flat_param
            if (
                param.dtype != torch.float32
                and param.device != torch.device("meta")
                and param.requires_grad
            ):
                param.data = param.data.to(torch.float32)
                m._handle._orig_param_dtype = torch.float32


def get_module_to_ignore_mixed_precision():
    try:
        from apex.normalization import FusedLayerNorm

        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            FusedLayerNorm,
        ]
    except:
        return [
            torch.nn.GroupNorm,
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
        ]


def is_fsdp_model(model):
    return len(FSDP.fsdp_modules(model)) > 0


def size_based_auto_wrap_policy(
    module: torch.nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # Additional custom arguments
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[torch.nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[torch.nn.Module]]] = None,
) -> bool:
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        min_num_params (int): Customizable policy input that controls the size
            threshold over which a module is ready to be wrapped. This is in
            units of numel.
        force_leaf_modules (Set[Type[nn.Module]]): Set of module types to keep
            as leaves, i.e. their children will never be wrapped.
        exclude_wrap_modules (Set[Type[nn.Module]]): Set of module types to be
            excluded in wrapping.

    Returns:
        Whether ``module`` should be wrapped.
    """
    force_leaf_modules = (
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None
        else force_leaf_modules
    )
    exclude_wrap_modules = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
        if exclude_wrap_modules is None
        else exclude_wrap_modules
    )

    # Keep the argument `min_num_params` for BC for now, but it represents the
    # minimum non-wrapped *numel* before triggering a wrapping
    min_nonwrapped_numel = min_num_params
    is_large = nonwrapped_numel >= min_nonwrapped_numel
    STOP_FLAG_NAME = "__FSDP_STOP_WARP_FLAG_CUSTOM_POLICY_size_based_auto_wrap_policy"
    if recurse:
        # use MixedPrecision cause ALWAYS recurse
        if isinstance(module, tuple(force_leaf_modules)):
            for m in module.children():
                m.apply(lambda m: setattr(m, STOP_FLAG_NAME, True))
        return True
    else:
        if getattr(module, size_based_auto_wrap_policy.LEAF_ROOT_FLAG_NAME, False):
            return True
        elif getattr(module, STOP_FLAG_NAME, False):
            return False
        else:
            # If we are not recursing, determine if we should wrap.
            return is_large and not isinstance(module, tuple(exclude_wrap_modules))


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {torch.nn.ModuleList, torch.nn.ModuleDict}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {torch.nn.MultiheadAttention}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.LEAF_ROOT_FLAG_NAME = (
    "__FSDP_LEAF_ROOT_FLAG_CUSTOM_POLICY_size_based_auto_wrap_policy"
)


def mark_leaf_root_(module):
    setattr(
        module,
        size_based_auto_wrap_policy.LEAF_ROOT_FLAG_NAME,
        True,
    )


def make_model_fsdp(
    model,
    param_dtype,
    device,
    reduce_dtype=None,
    buffer_dtype=None,
    sync_module_states=True,
    process_group=None,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    module_classes_to_ignore_mixed_precision=None,
    ignored_states=None,
    ignored_modules=None,
    auto_wrap_policy=None,
    part_size=1e6,
    force_leaf_modules=None,
    exclude_wrap_modules=None,
    use_orig_params=False
):
    if module_classes_to_ignore_mixed_precision is None:
        module_classes_to_ignore_mixed_precision = (
            get_module_to_ignore_mixed_precision()
        )
    if auto_wrap_policy is not None:
        auto_wrap_policy = auto_wrap_policy
    elif sharding_strategy == ShardingStrategy.NO_SHARD:
        auto_wrap_policy = None
    else:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=part_size,
            force_leaf_modules=force_leaf_modules,
            exclude_wrap_modules=exclude_wrap_modules,
        )

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        process_group=process_group,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype or torch.float32,
            buffer_dtype=buffer_dtype or torch.float32,
            keep_low_precision_grads=False,
            cast_forward_inputs=False,
            cast_root_forward_inputs=True,
            _module_classes_to_ignore=module_classes_to_ignore_mixed_precision,
        ),
        auto_wrap_policy=auto_wrap_policy,
        ignored_states=ignored_states,
        ignored_modules=ignored_modules,
        device_id=device,
    )
    torch.cuda.empty_cache()
    gc.collect()
    return model


def save_fsdp_model(
    model_to_save: FSDP,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
    max_shard_size: Union[int, str] = "10GB",
):
    unwrapped_model = accelerate.utils.extract_model_from_parallel(model_to_save)

    _REGEX_SHARD = re.compile(r"(.*?)-\d{5}-of-\d{5}")
    if isinstance(unwrapped_model, transformers.PreTrainedModel):
        _WEIGHTS_NAME = transformers.utils.SAFE_WEIGHTS_NAME
        _WEIGHTS_INDEX_FILE = transformers.utils.SAFE_WEIGHTS_INDEX_NAME
    else:
        _WEIGHTS_NAME = diffusers.utils.constants.SAFETENSORS_WEIGHTS_NAME
        _WEIGHTS_INDEX_FILE = diffusers.utils.constants.SAFE_WEIGHTS_INDEX_NAME

    weights_name = _WEIGHTS_NAME
    weight_name_split = weights_name.split(".")
    if len(weight_name_split) in [2, 3]:
        weights_name_pattern = (
            weight_name_split[0] + "{suffix}." + ".".join(weight_name_split[1:])
        )
    else:
        raise ValueError(f"Invalid {weights_name} provided.")

    os.makedirs(save_directory, exist_ok=True)

    # Save the config
    if is_main_process:
        if isinstance(unwrapped_model, transformers.PreTrainedModel):
            model_to_save.config.save_pretrained(save_directory)
        else:
            model_to_save.save_config(save_directory)

    # Save the model
    state_dict = torch_dcp.state_dict.get_model_state_dict(
        model_to_save,
        options=torch_dcp.state_dict.StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
        ),
    )

    # Save the model
    if is_main_process:
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            max_shard_size=max_shard_size,
            filename_pattern=weights_name_pattern,
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            if filename in state_dict_split.filename_to_tensors.keys():
                continue
            full_filename = os.path.join(save_directory, filename)
            if not os.path.isfile(full_filename):
                continue
            weights_without_ext = weights_name_pattern.replace(".bin", "").replace(
                ".safetensors", ""
            )
            weights_without_ext = weights_without_ext.replace("{suffix}", "")
            filename_without_ext = filename.replace(".bin", "").replace(
                ".safetensors", ""
            )
            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            if (
                filename.startswith(weights_without_ext)
                and _REGEX_SHARD.fullmatch(filename_without_ext) is not None
            ):
                os.remove(full_filename)

        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            filepath = os.path.join(save_directory, filename)
            safetensors.torch.save_file(shard, filepath, metadata={"format": "pt"})

        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = _WEIGHTS_INDEX_FILE
            save_index_file = os.path.join(save_directory, save_index_file)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)


def load_fsdp_model_(model_to_load: FSDP, save_directory: Union[str, os.PathLike]):
    with FSDP.state_dict_type(
        model_to_load,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(
            rank0_only=False,
        ),
    ):
        _model = model_to_load.from_pretrained(save_directory)
        model_to_load.load_state_dict(_model.state_dict())


def save_fsdp_optimizer(
    models: Dict,
    optimizer_to_save: torch.optim.Optimizer,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
):
    _fsdp_state_dict_config = dict(
        state_dict_type=StateDictType.FULL_STATE_DICT,
        optim_state_dict_config=FullOptimStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        ),
    )
    mgrs = list()
    for m in models.values():
        if len(FSDP.fsdp_modules(m)) > 0:
            mgrs.append(FSDP.state_dict_type(m, **_fsdp_state_dict_config))

    with contextlib.ExitStack() as stack:
        for mgr in mgrs:
            stack.enter_context(mgr)
        optim_state_dict = FSDP.optim_state_dict(
            torch.nn.ModuleDict(models),
            optimizer_to_save,
        )
        if is_main_process:
            torch.save(
                optim_state_dict, os.path.join(save_directory, "optim_states.pth")
            )


def load_fsdp_optimizer_(
    models: Dict,
    optimizer_to_load: torch.optim.Optimizer,
    save_directory: Union[str, os.PathLike],
):
    _fsdp_state_dict_config = dict(
        state_dict_type=StateDictType.FULL_STATE_DICT,
        optim_state_dict_config=FullOptimStateDictConfig(
            rank0_only=False,
        ),
    )
    mgrs = list()
    for m in models.values():
        if len(FSDP.fsdp_modules(m)) > 0:
            mgrs.append(FSDP.state_dict_type(m, **_fsdp_state_dict_config))

    with contextlib.ExitStack() as stack:
        for mgr in mgrs:
            stack.enter_context(mgr)
        optimizer_path = os.path.join(save_directory, "optim_states.pth")
        assert os.path.isfile(optimizer_path)
        optim_state_dict = torch.load(optimizer_path)
        optim_state_dict = FSDP.optim_state_dict_to_load(
            torch.nn.ModuleDict(models),
            optimizer_to_load,
            optim_state_dict,
        )
        optimizer_to_load.load_state_dict(optim_state_dict)


def save_fsdp_ema(
    ema_model: EMAModel,
    model: FSDP,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
):
    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())
    save_fsdp_model(model, save_directory, is_main_process=is_main_process)
    if is_main_process:
        state_dict = ema_model.state_dict()
        state_dict.pop("shadow_params", None)
        torch.save(state_dict, os.path.join(save_directory, "ema_states.pth"))
    ema_model.restore(model.parameters())


def load_fsdp_ema_(
    ema_model: EMAModel,
    model: FSDP,
    save_directory: Union[str, os.PathLike],
):
    ema_model.store(model.parameters())
    load_fsdp_model_(model, save_directory)
    ema_model.load_state_dict(
        torch.load(os.path.join(save_directory, "ema_states.pth"), weights_only=True)
    )
    ema_model.shadow_params = [p.clone().detach() for p in model.parameters()]
    ema_model.restore(model.parameters())
