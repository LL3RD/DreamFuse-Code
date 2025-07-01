import gc
import os

from typing import List
import contextlib
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from collections import defaultdict
import random
import numpy as np
from PIL import Image, ImageOps
import json
import torch
import dreamfuse.utils
import copy
import prodigyopt
from einops import rearrange
import yaml
from utils.inference_utils import find_nearest_bucket_size
import torch.nn.functional as F
# fmt: off
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    dreamfuse.utils.patch_npu_record_stream()
    dreamfuse.utils.patch_npu_diffusers_get_1d_rotary_pos_embed()
    USE_NPU = True
except:
    USE_NPU = False
# fmt: on
from torch.distributed.fsdp.api import ShardingStrategy
import accelerate
import diffusers
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
import transformers
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from dreamfuse.models.dreamfuse_flux.transformer import (
    FluxTransformer2DModel,
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from dreamfuse.utils import (
    remove_excess_checkpoints,
    import_from_transformers_modules,
    sync_to_remote,
    contain_invalid_grad,
)
from dreamfuse.fsdp_utils import (
    mark_leaf_root_,
    make_model_fsdp,
    get_module_to_ignore_mixed_precision,
    upcast_trainable_param_to_fp32_,
    save_fsdp_model,
    load_fsdp_model_,
    save_fsdp_optimizer,
    load_fsdp_optimizer_,
    save_fsdp_ema,
    load_fsdp_ema_,
    is_fsdp_model,
)
from dreamfuse.datasets.dreamfuse.merge_dataset import create_merged_dataset
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from dreamfuse.trains.utils.inference_utils import (
    compute_text_embeddings,
    prepare_latents,
    encode_images_cond,
    _unpack_latents,
    _pack_latents,
    _prepare_image_ids,
    get_mask_affine,
    warp_affine_tensor
)

@dataclass
class TrainConfig:
    flux_model_id: str = "exp_output/FLUX.1-dev/"
    model_choice: str = 'dev'
    work_mode: str = 'dreamfuse'
    output_dir: str = "exp_output/vlux_encoder"
    remote_output_dir: str = None
    data_config: str = "data/edit_data.yaml"
    revision: str = None
    variant: str = None
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    allow_tf32: bool = True
    resume_from_checkpoint: str = "latest"
    checkpoints_total_limit: int = 1
    save_model_steps: int = 1000
    val_interval: int = 500
    dataloader_num_workers: int = 8
    seed: int = 1020
    train_batch_size: int = 30
    num_training_steps: int = 100000
    lr_schedule_type: str = "cosine"
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    prodigy_beta3: float = None
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    prodigy_decouple: bool = True
    prodigy_use_bias_correction: bool = True
    prodigy_safeguard_warmup: bool = True
    prodigy_d0: float =1e-6
    lr_warmup_ratio: float = 0.02
    tune_denoise_dual_block_ids: List[int] = field(
        default_factory=lambda: tuple(range(19))
    )
    tune_denoise_single_block_ids: List[int] = field(
        default_factory=lambda: tuple(range(38))
    )
    tune_denoise_module: List[str] = field(
        default_factory=lambda: ("norm", "proj", "attn_norm", "attn_proj")
    )
    guidance_scale: float = 3.5
    max_sequence_length: int = 256
    image_resolution_aug_ratio: float = 0.2
    shard_text_model: bool = False
    shard_denoise_model: bool = False
    use_lora: bool = False
    lora_rank: int = 4
    valid_config: str = "./examples/valid/data_dreamfuse.json"
    valid_output_dir: str = "./valid_results"
    image_ids_offset: List[int] = None
    image_tags: List[int] = None
    context_tags: List[int] = None
    clip_grad_norm: float = 1.0
    mix_attention_double: bool = True
    mix_attention_single: bool = True
    valid_num_inference_steps: int = 28
    train_group_different_timestep: bool = False
    cat_by: str = 'batch'  # batch or length
    debug: bool = True
    ref_prompts: str = ""
    mask_tokenizer: bool = False
    mask_in_chans: int = 128
    mask_out_chans: int = 3072
    mask_ids: int = 16

    mask_pos_affine: bool = True
    prompt_ratio: float = 0.01

    load_from_checkpoint: str = None


def make_image_grid(images, rows, cols, resize=None):
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img.convert("RGB"), box=(i % cols * w, i // cols * h))
    return grid


def read_yaml(path):
    with open(path) as fr:
        return yaml.safe_load(fr)



@torch.no_grad()
def log_validation(
        config,
        denoise_model,
        vae_model,
        text_encoders,
        tokenizers,
        accelerator,
        scheduler,
        global_step=None,
        guidance_scale=3.5,
        num_channels_latents=16,
        sigmas=None,
        latents=None,
):
    num_inference_steps = config.valid_num_inference_steps
    data_infos = read_yaml(config.data_config)['datasets']

    for data_info in data_infos:
        if "valid_root" not in data_info:
            continue
        
        if isinstance(data_info['valid_json'], str):
            with open(data_info['valid_json'], 'r') as f:
                valid_info = json.load(f)
        elif isinstance(data_info['valid_json'], list):
            valid_info = {}
            for valid_json in data_info['valid_json']:
                with open(valid_json, 'r') as f:
                    valid_info.update(json.load(f))
        
        for idx, (meta_key, info) in enumerate(valid_info.items()):            
            sorted_keys = info["sorted_keys"]
            images_cond = [
                Image.open(os.path.join(data_info["valid_root"], info['img_info']['000'])),
                Image.open(os.path.join(data_info["valid_root"], info['img_info']['001'])),
            ]
            masks_cond = [
                Image.open(os.path.join(data_info["valid_root"], info['img_mask_info']['000_mask_scale'])),
                Image.open(os.path.join(data_info["valid_root"], info['img_mask_info']['000'])),
                Image.open(os.path.join(data_info["valid_root"], info['img_mask_info']['000_paste'])),
            ]


            images_cond[0].paste((255, 255, 255), mask=ImageOps.invert(masks_cond[1]))
            
            if images_cond[0].size[0] % 16 != 0 or images_cond[0].size[1] % 16 != 0:
                images_cond = [img.resize((images_cond[0].size[0] - images_cond[0].size[0] % 16, images_cond[0].size[1] - images_cond[0].size[1] % 16)) for img in images_cond]
            
            bucket_size = find_nearest_bucket_size(images_cond[0].size[0], images_cond[0].size[1])
            images_cond = [img.resize(bucket_size) for img in images_cond]
            images_cond = [img.convert("RGB") for img in images_cond]
            masks_cond = [img.resize(bucket_size) for img in masks_cond]

            if config.mask_pos_affine:
                mask_affines = [get_mask_affine(im[0], im[1]) for im in [masks_cond]]

            # prompts_ = [[info['caption_info']['002']], [config.ref_prompts]][-1:]
            prompts_ = [[config.ref_prompts]]
            for p_idx, prompts in enumerate(prompts_):
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    config, prompts, text_encoders, tokenizers, accelerator.device
                )

                batch_size = len(prompts)
                data_num_per_group = len(prompts)
                width, height = images_cond[0].size

                # handle guidance
                if config.model_choice == 'dev':  # denoise_model.config.guidance_embeds, but ddp need unwarp
                    guidance = torch.full(
                        [1], guidance_scale, device=accelerator.device, dtype=torch.float32
                    )
                    guidance = guidance.expand(batch_size)
                else:
                    guidance = None

                generator = [
                    torch.Generator(device=accelerator.device).manual_seed(config.seed + i) for i in range(batch_size)
                ]

                # 4.0 Prepare Condition latent variables
                condition_image_latents = encode_images_cond(vae_model, [images_cond[:-1]], accelerator.device) # shape: [bs*n_cond, c, h // 8, w // 8]
                # 4.1 Prepare Condition latent & ids
                offset_cond = config.image_ids_offset[1:]
                cond_latent_image_ids = []
                for offset_ in offset_cond:
                    cond_latent_image_ids.append(
                        _prepare_image_ids(
                            condition_image_latents.shape[2] // 2, condition_image_latents.shape[3] // 2, offset_w=offset_ *condition_image_latents.shape[3] // 2
                        ).to(device=accelerator.device).float()
                    )

                if config.mask_pos_affine:
                    affine_H, affine_W = condition_image_latents.shape[2] // 2, condition_image_latents.shape[3] // 2
                    scale_factor = 1 / 16
                    cond_latent_image_ids_fg = cond_latent_image_ids[0].reshape(affine_H, affine_W, 3).clone()
                    # Affine whole image
                    cond_latent_image_ids[0] = warp_affine_tensor(
                        cond_latent_image_ids_fg, mask_affines[0], output_size=(affine_H, affine_W),
                        scale_factor=scale_factor, device=accelerator.device, flatten_output=True
                    )      
                
                cond_latent_image_ids = torch.stack(cond_latent_image_ids)
                cond_image_latents = _pack_latents(condition_image_latents, *condition_image_latents.shape)
                
                cond_input = {
                    "image_latents": cond_image_latents,
                    "image_ids": cond_latent_image_ids,
                } 

                if config.mask_tokenizer:
                    mask_ids = _prepare_image_ids(config.mask_ids, config.mask_ids).to(device=accelerator.device).float()

                    # 4.2 Prepare mask cond
                    mask_image_tensors = [torch.tensor(np.array(_)) for _ in masks_cond[:1]]
                    mask_image_tensors = torch.stack(mask_image_tensors).to(accelerator.device) / 255.0
                    mask_image_tensors = F.interpolate(mask_image_tensors.unsqueeze(1).float(), size=(
                        condition_image_latents.shape[2], condition_image_latents.shape[3]
                    ), mode='bilinear')
                    mask_image_tensors = mask_image_tensors > 0.5
                    mask_image_tensors = mask_image_tensors.to(condition_image_latents.dtype)

                # 4. Prepare latent variables
                vae_scale_factor = (
                        2 ** (len(vae_model.config.block_out_channels) - 1)
                )
                latents, latent_image_ids = prepare_latents(
                    batch_size,
                    num_channels_latents,
                    vae_scale_factor,
                    height,
                    width,
                    prompt_embeds.dtype,
                    accelerator.device,
                    generator,
                    None,
                    None,
                )

                # 5. Prepare timesteps
                sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
                image_seq_len = latents.shape[1]
                mu = calculate_shift(
                    image_seq_len,
                    scheduler.config.base_image_seq_len,
                    scheduler.config.max_image_seq_len,
                    scheduler.config.base_shift,
                    scheduler.config.max_shift,
                )
                timesteps, num_inference_steps = retrieve_timesteps(
                    scheduler,
                    num_inference_steps,
                    accelerator.device,
                    sigmas=sigmas,
                    mu=mu,
                )

                # 6. Denoising loop
                for i, t in enumerate(timesteps):
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    if config.mask_tokenizer:
                        joint_attention_kwargs = {"work_mode": config.work_mode, "mask_cond": mask_image_tensors, "mask_ids": mask_ids}
                    else:
                        joint_attention_kwargs = None

                    with torch.autocast(
                            enabled=True, device_type="cuda", dtype=config.weight_dtype
                    ):
                        noise_pred = denoise_model(
                            hidden_states=latents,
                            cond_input=cond_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            data_num_per_group=data_num_per_group,
                            image_tags=config.image_tags,
                            context_tags=config.context_tags,
                            max_sequence_length=config.max_sequence_length,
                            mix_attention_double=config.mix_attention_double,
                            mix_attention_single=config.mix_attention_single,
                            joint_attention_kwargs=joint_attention_kwargs,
                            return_dict=False,
                        )[0]

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                latents = _unpack_latents(latents, height, width, vae_scale_factor)
                latents = (latents / vae_model.config.scaling_factor) + vae_model.config.shift_factor
                images = vae_model.decode(latents.to(latents_dtype), return_dict=False)[0]
                images = (
                    images.add(1)
                    .mul(127.5)
                    .clamp(0, 255)
                    .to(torch.int8)
                    .permute([0, 2, 3, 1])
                    .cpu()
                    .numpy()
                )
                images = images_cond + [Image.fromarray(_, "RGB") for _ in images] + masks_cond[-1:]

                if accelerator.is_main_process:
                    save_dir = os.path.join(config.valid_output_dir, str(global_step))
                    os.makedirs(save_dir, exist_ok=True)
                    res = make_image_grid(images, 1, len(images))
                    res.save(os.path.join(save_dir, meta_key + f"_{p_idx}.jpg"))


def main():
    parser = transformers.HfArgumentParser(TrainConfig)
    config: TrainConfig = parser.parse_args_into_dataclasses()[0]
    if config.debug:
        update_default_config(config) # for debug
    else:
        if config.image_tags and config.image_tags[0] == -1:
            config.image_tags = None
        if config.context_tags and config.context_tags[0] == -1:
            config.context_tags = None
    
    # assert config.work_mode in ('dreamfuse')

    accelerator = accelerate.Accelerator(
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    config.weight_dtype = torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        config.weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        config.weight_dtype = torch.bfloat16

    if config.seed is not None:
        accelerate.utils.set_seed(config.seed)
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process and config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("smartedit-aes2.0")

    # Initialize text encoder
    # Load the tokenizers
    tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
        config.flux_model_id,
        subfolder="tokenizer",
        revision=config.revision,
    )
    tokenizer_two = transformers.T5TokenizerFast.from_pretrained(
        config.flux_model_id,
        subfolder="tokenizer_2",
        revision=config.revision,
    )

    # import correct text encoder classes
    text_encoder_one = transformers.CLIPTextModel.from_pretrained(
        config.flux_model_id,
        subfolder="text_encoder",
        revision=config.revision,
        variant=config.variant
    )
    text_encoder_two = transformers.T5EncoderModel.from_pretrained(
        config.flux_model_id,
        subfolder="text_encoder_2",
        revision=config.revision,
        variant=config.variant
    )
    text_encoder_one.train(False).requires_grad_(False)
    text_encoder_two.train(False).requires_grad_(False)

    # Initialize vae model
    vae_model = (
        diffusers.AutoencoderKL.from_pretrained(
            config.flux_model_id, subfolder="vae", torch_dtype=config.weight_dtype
        )
        .to(accelerator.device)
        .train(False)
        .requires_grad_(False)
    )

    vae_scale_factor = (
            2 ** (len(vae_model.config.block_out_channels) - 1)
    )

    # Initialize noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.flux_model_id, subfolder="scheduler"
    )
    val_noise_scheduler = copy.deepcopy(noise_scheduler)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Initialize denoise model
    denoise_model = (
        FluxTransformer2DModel.from_pretrained(
            config.flux_model_id, subfolder="transformer"
        )
        .train(True)
        .requires_grad_(not config.use_lora)
    )
    if config.image_tags is not None or config.context_tags is not None:
        num_image_tag_embeddings = max(config.image_tags) + 1 if config.image_tags is not None else 0
        num_context_tag_embeddings = max(config.context_tags) + 1 if config.context_tags is not None else 0
        denoise_model.set_tag_embeddings(
            num_image_tag_embeddings=num_image_tag_embeddings,
            num_context_tag_embeddings=num_context_tag_embeddings
        )
    if config.mask_tokenizer:
        denoise_model.set_mask_tokenizer(
            config.mask_in_chans, config.mask_out_chans
        )
    if config.gradient_checkpointing:
        denoise_model.enable_gradient_checkpointing()

    if config.use_lora:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        if config.image_tags is not None or config.context_tags is not None or config.mask_tokenizer:
            modules_to_save_ = []
            if config.image_tags is not None:
                modules_to_save_.append('image_tag_embeddings')
            if config.context_tags is not None:
                modules_to_save_.append('context_tag_embeddings')
            if config.mask_tokenizer:
                modules_to_save_.append('mask_tokenizer')
                modules_to_save_.append('mask_attn')

            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
                modules_to_save=modules_to_save_
            )
        else:
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )

        # denoise_model.add_adapter(lora_config)
        denoise_model = get_peft_model(denoise_model, lora_config)
        if config.load_from_checkpoint is not None:
            denoise_model.load_adapter(
                    os.path.join(config.load_from_checkpoint, "denoise_model"), 'default', is_trainable=True,
                    device_map={"": accelerator.device}
                )
        denoise_model.print_trainable_parameters()

    if config.shard_text_model:
        text_encoder_two = make_model_fsdp(
            text_encoder_two,
            param_dtype=config.weight_dtype,
            device=accelerator.device,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            sync_module_states=True,
            part_size=1e5,
            force_leaf_modules=(
                torch.nn.Embedding,
                torch.nn.TransformerDecoderLayer,
                torch.nn.TransformerEncoderLayer,
            ),
        )
        accelerator.print(
            "text_encoder_two Root FSDP module",
            torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules(
                text_encoder_two, root_only=True
            ),
        )
    else:
        text_encoder_two = text_encoder_two.to(accelerator.device, config.weight_dtype)
    text_encoder_one = text_encoder_one.to(accelerator.device, config.weight_dtype)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]
    num_channels_latents = denoise_model.config.in_channels // 4
    if config.shard_denoise_model:
        denoise_model = make_model_fsdp(
            denoise_model,
            param_dtype=config.weight_dtype,
            device=accelerator.device,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            sync_module_states=False,
            part_size=1e5,
            force_leaf_modules=(
                FluxTransformerBlock,
                FluxSingleTransformerBlock,
            ),
            use_orig_params=config.use_lora,
        )
        upcast_trainable_param_to_fp32_(denoise_model)
        accelerator.print(
            "denoise_model Root FSDP module",
            torch.distributed.fsdp.FullyShardedDataParallel.fsdp_modules(
                denoise_model, root_only=True
            ),
        )
    else:
        denoise_model = denoise_model.to(accelerator.device, config.weight_dtype)
        if accelerator.num_processes > 1:
            denoise_model = torch.nn.parallel.DistributedDataParallel(
                denoise_model, find_unused_parameters=config.find_unused_parameters
            )

    # Creates Optimizer
    # workaround resume bug in pytorch
    _optim_params = []
    _optim_params.append(
        {"params": denoise_model.parameters(), "lr": config.learning_rate}
    )
    if config.optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params=_optim_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )
    elif config.optimizer_type.lower() == "prodigy":
        optimizer = prodigyopt.Prodigy(
            params=_optim_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            beta3=config.prodigy_beta3,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
            decouple=config.prodigy_decouple,
            use_bias_correction=config.prodigy_use_bias_correction,
            safeguard_warmup=config.prodigy_safeguard_warmup,
            d0=config.prodigy_d0
        )

    # Creates learning rate scheduler
    lr_schedule_func = {
        "linear": transformers.get_linear_schedule_with_warmup,
        "cosine": transformers.get_cosine_schedule_with_warmup,
        "constant": transformers.get_constant_schedule_with_warmup,
    }[config.lr_schedule_type]
    if config.lr_schedule_type == 'constant':
        lr_scheduler = lr_schedule_func(
            optimizer=optimizer,
            num_warmup_steps=int(config.lr_warmup_ratio * config.num_training_steps),
        )
    else:
        lr_scheduler = lr_schedule_func(
            optimizer=optimizer,
            num_warmup_steps=int(config.lr_warmup_ratio * config.num_training_steps),
            num_training_steps=config.num_training_steps,
        )

    global_epoch = 0
    global_step = 0
    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[2]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            save_path = os.path.join(config.output_dir, path)
            random.setstate(
                torch.load(
                    os.path.join(save_path, "misc_states.pth"), weights_only=False
                )["random_state"]
            )
            lr_scheduler.load_state_dict(
                torch.load(
                    os.path.join(save_path, "lr_scheduler_states.pth"),
                    weights_only=False,
                )
            )

            denoise_model.load_adapter(
                os.path.join(save_path, "denoise_model"), 'default', is_trainable=True,
                device_map={"": accelerator.device}
            )

            torch.cuda.empty_cache()
            gc.collect()
            global_epoch = int(path.split("-")[1])
            global_step = int(path.split("-")[2])

    # build edit dataloader
    train_dataset = create_merged_dataset(
        config.data_config,
        batchsize=config.train_batch_size,
        rank=accelerator.process_index,
        worldsize=accelerator.num_processes,
        use_ratio=False,
    )
    accelerator.print(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=config.dataloader_num_workers,
        prefetch_factor=16,
        pin_memory=False,
        collate_fn=None,
    )

    num_step_per_epoch = (
            len(train_dataset)
            // config.train_batch_size
            // config.gradient_accumulation_steps
    )
    num_step_per_epoch = int(num_step_per_epoch)
    progress_bar = tqdm(
        total=config.num_training_steps,
        disable=not accelerator.is_local_main_process,
        initial=global_step % (config.num_training_steps + 1),
    )
    accelerator.wait_for_everyone()
    train_dataloader_iter = iter(train_dataloader)
    while global_step < config.num_training_steps:
        for _local_step in range(config.gradient_accumulation_steps):
            if (
                    _local_step == 0
                    and global_step > 0
                    and global_step % num_step_per_epoch == 0
            ):
                global_epoch += 1
                train_dataset.reset_local_meta_list_()
                accelerator.wait_for_everyone()

            data_batch = next(train_dataloader_iter)
            prompts, target_images, condition_images, mask_images = list(), list(), list(), list()
            if config.work_mode == 'dreamfuse':
                data_num_per_group = len(data_batch["prompts"])
                for idx_in_group in range(data_num_per_group):
                    prompt_prob = random.random()
                    if prompt_prob < config.prompt_ratio:
                        prompts += [p[-1] for p in data_batch["prompts"]]
                    else:
                        prompts += [config.ref_prompts for p in data_batch["prompts"]]
                    target_images += [im[-1] for im in data_batch["images"]]
                    condition_images += [im[:-1] for im in data_batch["images"]]
                if config.mask_tokenizer:
                    mask_images += [im[0] for im in data_batch["mask_images"]]
                else:
                    mask_images += [im[:2] for im in data_batch["mask_images"]]
            else:
                raise NotImplementedError

            with torch.no_grad():
                batchsize = len(prompts)  # output image num
                target_image_tensors = [
                    torch.tensor(np.array(_)) for _ in target_images
                ]
                target_image_tensors = (
                    torch.stack(target_image_tensors)
                    .to(accelerator.device)
                    .permute(0, 3, 1, 2)
                )
                target_image_tensors = target_image_tensors / 127.5 - 1.0
                
                #
                if len(condition_images) > 0:
                    condition_image_latents = encode_images_cond(vae_model, condition_images, accelerator.device)
                else:
                    condition_image_latents = None

                # prepare for diffusion model
                clean_output_image_latent = (
                    vae_model.encode(target_image_tensors.to(vae_model.dtype))
                    .latent_dist.sample()
                )
                clean_output_image_latent = (
                                                    clean_output_image_latent - vae_model.config.shift_factor
                                            ) * vae_model.config.scaling_factor

                if len(mask_images) and config.mask_tokenizer:
                    mask_image_tensors = [torch.tensor(np.array(_)) for _ in mask_images]
                    mask_image_tensors = torch.stack(mask_image_tensors).to(accelerator.device) / 255.0
                    mask_image_tensors = F.interpolate(mask_image_tensors.unsqueeze(1).float(), size=(
                        clean_output_image_latent.shape[2], clean_output_image_latent.shape[3]
                    ), mode='bilinear')
                    mask_image_tensors = mask_image_tensors > 0.5
                    mask_image_tensors = mask_image_tensors.to(clean_output_image_latent.dtype)

                    mask_ids = _prepare_image_ids(config.mask_ids, config.mask_ids).to(device=accelerator.device).float()
                elif len(mask_images) and config.mask_pos_affine:
                    # affine ids
                    mask_affines = [get_mask_affine(im[0], im[1]) for im in mask_images]
                else:
                    mask_affines = None
                    mask_image_tensors = None
                    mask_ids = None

                # add noise
                timesteps = noise_scheduler.timesteps[
                    torch.randint(0, noise_scheduler.timesteps.numel(), (batchsize,))
                ].to(accelerator.device)
                if data_num_per_group > 1 and (not config.train_group_different_timestep):
                    bsz = batchsize // data_num_per_group
                    timesteps = timesteps[:bsz].repeat(data_num_per_group)

                noise = torch.randn_like(clean_output_image_latent)
                # flow matching loss
                denoise_target = noise - clean_output_image_latent
                # prepare denoise model input
                noisy_latent = noise_scheduler.scale_noise(
                    clean_output_image_latent, timesteps, noise
                )

                # pack latents and prepare image ids
                if isinstance(config.image_ids_offset, list) and config.work_mode == 'dreamfuse':
                    offset_ = config.image_ids_offset[0]
                    latent_image_ids = _prepare_image_ids(
                                noisy_latent.shape[2] // 2, noisy_latent.shape[3] // 2, offset_w=offset_ * noisy_latent.shape[3] // 2
                            ).to(device=accelerator.device).float()
                    offset_cond = config.image_ids_offset[1:]
                    cond_latent_image_ids = []
                    for offset_ in offset_cond:
                        cond_latent_image_ids.append(
                            _prepare_image_ids(
                                noisy_latent.shape[2] // 2, noisy_latent.shape[3] // 2, offset_w=offset_ * noisy_latent.shape[3] // 2
                            ).to(device=accelerator.device).float()
                        )
                    
                    # mask_pos_affine
                    if config.mask_pos_affine:
                        affine_H, affine_W = noisy_latent.shape[2] // 2, noisy_latent.shape[3] // 2
                        scale_factor = 1 / 16
                        cond_latent_image_ids_fg = cond_latent_image_ids[0].reshape(affine_H, affine_W, 3).clone()

                        # Todo: support bsz > 1
                        cond_latent_image_ids[0] = warp_affine_tensor(
                            cond_latent_image_ids_fg, mask_affines[0], output_size=(affine_H, affine_W),
                            scale_factor=scale_factor, device=accelerator.device,
                        )

                        
                    cond_latent_image_ids = torch.stack(cond_latent_image_ids)
                    cond_image_latents = _pack_latents(condition_image_latents, *condition_image_latents.shape)
                    cond_input = {
                        "image_latents": cond_image_latents,
                        "image_ids": cond_latent_image_ids,
                    }
                elif isinstance(config.image_ids_offset, list):
                    latent_image_ids = []
                    for offset_ in config.image_ids_offset:
                        latent_image_ids.append(
                            _prepare_image_ids(
                                noisy_latent.shape[2] // 2, noisy_latent.shape[3] // 2, offset_w=offset_ * noisy_latent.shape[3] // 2
                            ).to(device=accelerator.device).float()
                        )
                    cond_input = None
                else:
                    latent_image_ids = _prepare_image_ids(
                        noisy_latent.shape[2] // 2, noisy_latent.shape[3] // 2,
                    )
                    latent_image_ids = latent_image_ids.to(device=accelerator.device).float()
                    cond_input = None

                noisy_latent = _pack_latents(
                    noisy_latent, *noisy_latent.shape
                )

            prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                config, prompts, text_encoders, tokenizers, accelerator.device
            )

            if config.mask_tokenizer:
                joint_attention_kwargs = {"work_mode": config.work_mode, "mask_cond": mask_image_tensors, "mask_ids": mask_ids}
            else:
                joint_attention_kwargs = None
                
            with torch.autocast(
                    enabled=True, device_type="cuda", dtype=config.weight_dtype
            ):
                denoise_output = denoise_model(
                    hidden_states=noisy_latent,
                    cond_input=cond_input,
                    timestep=timesteps.float() / 1000,
                    guidance=torch.full(
                        [batchsize], fill_value=config.guidance_scale, device=accelerator.device
                    ).float() if config.model_choice == 'dev' else None,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    data_num_per_group=data_num_per_group,
                    image_tags=config.image_tags,
                    context_tags=config.context_tags,
                    max_sequence_length=config.max_sequence_length,
                    mix_attention_double=config.mix_attention_double,
                    mix_attention_single=config.mix_attention_single,
                    joint_attention_kwargs=joint_attention_kwargs
                ).sample

            denoise_output = _unpack_latents(
                denoise_output,
                height=int(clean_output_image_latent.shape[2] * vae_scale_factor),
                width=int(clean_output_image_latent.shape[3] * vae_scale_factor),
                vae_downsample_factor=vae_scale_factor,
            )

            loss = torch.nn.functional.mse_loss(
                denoise_output.float(), denoise_target.float(), reduction="mean"
            )
            loss = loss / config.gradient_accumulation_steps

            with torch.no_grad():
                _low_step_loss = (denoise_output - denoise_target).pow(2).mean([2, 3])
                _low_step_loss = _low_step_loss[timesteps <= 200].mean().detach()

            with contextlib.ExitStack() as stack:
                if _local_step < config.gradient_accumulation_steps - 1:
                    stack.enter_context(denoise_model.no_sync())
                loss.backward()
            if contain_invalid_grad(optimizer):
                optimizer.zero_grad(set_to_none=True)
                accelerator.print(f"grad nan skip", flush=True)
                break
            if _local_step == config.gradient_accumulation_steps - 1:
                if config.clip_grad_norm > 0:
                    if config.shard_denoise_model:
                        denoise_model.clip_grad_norm_(config.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), config.clip_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=False)
        else:
            global_step += 1
            progress_bar.update(1)
            logs = {
                f"loss": loss.detach().item()
                         * config.gradient_accumulation_steps,
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if not torch.isnan(_low_step_loss).any():
                logs["low_step_loss"] = _low_step_loss.item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                save_path = os.path.join(
                    config.output_dir,
                    f"checkpoint-{global_epoch:05}-{global_step:08}",
                )
                os.makedirs(save_path, exist_ok=True)
                if accelerator.is_main_process:
                    torch.save(
                        dict(random_state=random.getstate()),
                        os.path.join(save_path, "misc_states.pth"),
                    )
                    torch.save(
                        lr_scheduler.state_dict(),
                        os.path.join(save_path, "lr_scheduler_states.pth"),
                    )

                _save_models = dict()
                _save_models["denoise_model"] = denoise_model
                if not config.use_lora:
                    # hehehehe
                    save_fsdp_optimizer(
                        _save_models,
                        optimizer,
                        save_path,
                        is_main_process=accelerator.is_main_process,
                    )
                for n, m in _save_models.items():
                    if is_fsdp_model(m):
                        save_fsdp_model(
                            m,
                            os.path.join(save_path, n),
                            is_main_process=accelerator.is_main_process,
                        )
                    else:
                        if accelerator.is_main_process:
                            unwrap_model(m).save_pretrained(os.path.join(save_path, n))

                torch.cuda.empty_cache()
                gc.collect()
                accelerator.print(f"Saved state to {save_path}")
                remove_excess_checkpoints(
                    config.output_dir,
                    config.checkpoints_total_limit,
                    checkpoint_prefix="checkpoint",
                    is_main_process=accelerator.is_main_process,
                )
                if config.remote_output_dir and accelerator.is_main_process:
                    sync_to_remote(
                        config.output_dir, config.remote_output_dir, blocking=False
                    )
                accelerator.wait_for_everyone()

            if global_step % config.val_interval == 0 or global_step == 1:
                accelerator.wait_for_everyone()
                # ema - preprocess
                log_validation(
                    config,
                    denoise_model,
                    vae_model,
                    text_encoders,
                    tokenizers,
                    accelerator,
                    val_noise_scheduler,
                    global_step=global_step,
                    num_channels_latents=num_channels_latents
                )
    save_path = config.output_dir
    _save_models = dict()
    _save_models["denoise_model"] = denoise_model
    for n, m in _save_models.items():
        if is_fsdp_model(m):
            save_fsdp_model(
                m,
                os.path.join(save_path, n),
                is_main_process=accelerator.is_main_process,
            )
        else:
            if accelerator.is_main_process:
                unwrap_model(m).save_pretrained(os.path.join(save_path, n))
    if config.remote_output_dir and accelerator.is_main_process:
        sync_to_remote(config.output_dir, config.remote_output_dir, blocking=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    if not USE_NPU:
        mp.set_start_method("spawn", force=True)
    main()
