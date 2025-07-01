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
from peft import PeftModel
import torch.nn.functional as F
import accelerate
import diffusers
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
import transformers
from tqdm import tqdm
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from dreamfuse.models.dreamfuse_flux.transformer import (
    FluxTransformer2DModel,
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from dreamfuse.trains.utils.inference_utils import (
    compute_text_embeddings,
    prepare_latents,
    _unpack_latents,
    _pack_latents,
    _prepare_image_ids,
    encode_images_cond,
    get_mask_affine,
    warp_affine_tensor
)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

@dataclass
class InferenceConfig:
    # Model paths
    flux_model_id: str = 'black-forest-labs/FLUX.1-dev'
    
    lora_id: str = 'LL3RD/DreamFuse'
    model_choice: str = 'dev'
    # Model configs
    lora_rank: int = 16
    max_sequence_length: int = 256
    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    mask_ids: int = 16
    mask_in_chans: int = 128 
    mask_out_chans: int = 3072
    inference_scale = 1024
    
    # Training configs
    gradient_checkpointing: bool = False
    mix_attention_double: bool = True
    mix_attention_single: bool = True
    
    # Image processing
    image_ids_offset: List[int] = field(default_factory=lambda: [0, 0, 0])
    image_tags: List[int] = field(default_factory=lambda: [0, 1, 2])
    context_tags: List[int] = None
    
    # Runtime configs
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    seed: int = 1234
    debug: bool = True

    # I/O configs
    valid_output_dir: str = "./inference_output"
    valid_roots: List[str] = field(default_factory=lambda: [
        "./examples",
    ])
    valid_jsons: List[str] = field(default_factory=lambda: [
        "./examples/data_dreamfuse.json",
    ])
    ref_prompts: str = ""
    
    truecfg: bool = False
    text_strength: int = 5
    
    # multi gpu
    sub_idx: int = 0
    total_num: int = 1

def adjust_fg_to_bg(image: Image.Image, mask: Image.Image, target_size: tuple) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    target_w, target_h = target_size
    
    scale = min(target_w / width, target_h / height)
    if scale < 1:
        new_w = int(width * scale)
        new_h = int(height * scale)
        image = image.resize((new_w, new_h))
        mask = mask.resize((new_w, new_h))
        width, height = new_w, new_h
    
    pad_w = target_w - width
    pad_h = target_h - height
    padding = (
        pad_w // 2,  # left
        pad_h // 2,  # top 
        (pad_w + 1) // 2,  # right
        (pad_h + 1) // 2   # bottom
    )
    
    image = ImageOps.expand(image, border=padding, fill=(255, 255, 255))
    mask = ImageOps.expand(mask, border=padding, fill=0)
    
    return image, mask

def find_nearest_bucket_size(input_width, input_height, mode="x64", bucket_size=1024):
    """
    Finds the nearest bucket size for the given input size.
    """
    buckets = {
        512: [[ 256, 768 ], [ 320, 768 ], [ 320, 704 ], [ 384, 640 ], [ 448, 576 ], [ 512, 512 ], [ 576, 448 ], [ 640, 384 ], [ 704, 320 ], [ 768, 320 ], [ 768, 256 ]],
        768: [[ 384, 1152 ], [ 480, 1152 ], [ 480, 1056 ], [ 576, 960 ], [ 672, 864 ], [ 768, 768 ], [ 864, 672 ], [ 960, 576 ], [ 1056, 480 ], [ 1152, 480 ], [ 1152, 384 ]],
        1024: [[ 512, 1536 ], [ 640, 1536 ], [ 640, 1408 ], [ 768, 1280 ], [ 896, 1152 ], [ 1024, 1024 ], [ 1152, 896 ], [ 1280, 768 ], [ 1408, 640 ], [ 1536, 640 ], [ 1536, 512 ]]
    }
    
    buckets = buckets[bucket_size]

    aspect_ratios = [w / h for (w, h) in buckets]
    assert mode in ["x64", "x8"]
    if mode == "x64":
        asp = input_width / input_height
        diff = [abs(ar - asp) for ar in aspect_ratios]
        bucket_id = int(np.argmin(diff))
        gen_width, gen_height = buckets[bucket_id]
    elif mode == "x8":
        max_pixels = 1024 * 1024
        ratio = (max_pixels / (input_width * input_height)) ** (0.5)
        gen_width, gen_height = round(input_width * ratio), round(input_height * ratio)
        gen_width = gen_width - gen_width % 8
        gen_height = gen_height - gen_height % 8
    else:
        raise NotImplementedError
    return (gen_width, gen_height)

def make_image_grid(images, rows, cols, size=None):
    assert len(images) == rows * cols

    if size is not None:
        images = [img.resize((size[0], size[1])) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img.convert("RGB"), box=(i % cols * w, i // cols * h))
    return grid

class DreamFuseInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        seed_everything(config.seed)
        self._init_models()

    def _init_models(self):
        # Initialize tokenizers
        self.tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
            self.config.flux_model_id, subfolder="tokenizer"
        )
        self.tokenizer_two = transformers.T5TokenizerFast.from_pretrained(
            self.config.flux_model_id, subfolder="tokenizer_2"
        )

        # Initialize text encoders
        self.text_encoder_one = transformers.CLIPTextModel.from_pretrained(
            self.config.flux_model_id, subfolder="text_encoder"
        ).to(device=self.device, dtype=self.config.dtype)
        self.text_encoder_two = transformers.T5EncoderModel.from_pretrained(
            self.config.flux_model_id, subfolder="text_encoder_2"
        ).to(device=self.device, dtype=self.config.dtype)

        # Initialize VAE
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            self.config.flux_model_id, subfolder="vae"
        ).to(device=self.device, dtype=self.config.dtype)

        # Initialize denoising model
        self.denoise_model = FluxTransformer2DModel.from_pretrained(
            self.config.flux_model_id, subfolder="transformer"
        ).to(device=self.device, dtype=self.config.dtype)

        if self.config.image_tags is not None or self.config.context_tags is not None:
            num_image_tag_embeddings = max(self.config.image_tags) + 1 if self.config.image_tags is not None else 0
            num_context_tag_embeddings = max(self.config.context_tags) + 1 if self.config.context_tags is not None else 0
            self.denoise_model.set_tag_embeddings(
                num_image_tag_embeddings=num_image_tag_embeddings,
                num_context_tag_embeddings=num_context_tag_embeddings,
            )

        # Add LoRA
        self.denoise_model = PeftModel.from_pretrained(
            self.denoise_model,
            self.config.lora_id,
            adapter_weights=[1.0],
            device_map={"": self.device}
        )

        # Initialize scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.flux_model_id, subfolder="scheduler"
        )

        # Set models to eval mode
        for model in [self.text_encoder_one, self.text_encoder_two, self.vae, self.denoise_model]:
            model.eval()
            model.requires_grad_(False)

    def _compute_text_embeddings(self, prompt):
        return compute_text_embeddings(
            self.config,
            prompt,
            [self.text_encoder_one, self.text_encoder_two],
            [self.tokenizer_one, self.tokenizer_two],
            self.device
        )

    @torch.no_grad()
    def __call__(self, fg_image, bg_image, ori_fg_mask, new_fg_mask, enable_mask_affine=True, prompt="", offset_cond=None, seed=None, cfg=3.5, size_select=1024, text_strength=1, truecfg=False):
        batch_size = 1
        
        # Prepare images
        # adjust bg->fg size
        fg_image, ori_fg_mask = adjust_fg_to_bg(fg_image, ori_fg_mask, bg_image.size)
        bucket_size = find_nearest_bucket_size(bg_image.size[0], bg_image.size[1], bucket_size=size_select)

        fg_image = fg_image.resize(bucket_size)
        bg_image = bg_image.resize(bucket_size)

        mask_affine = None
        if enable_mask_affine:
            ori_fg_mask = ori_fg_mask.resize(bucket_size)
            new_fg_mask = new_fg_mask.resize(bucket_size)
            mask_affine = get_mask_affine(new_fg_mask, ori_fg_mask)

        # Get embeddings
        prompt_embeds, pooled_prompt_embeds, text_ids = self._compute_text_embeddings(prompt)

        prompt_embeds = prompt_embeds.repeat(1, text_strength, 1)
        text_ids = text_ids.repeat(text_strength, 1)

        # Prepare 
        if self.config.model_choice == "dev":
            guidance = torch.full([1], cfg, device=self.device, dtype=torch.float32)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None

        # Prepare generator
        if seed is None:
            seed = self.config.seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Prepare condition latents
        condition_image_latents = self._encode_images([fg_image, bg_image])
        
        if offset_cond is None:
            offset_cond = self.config.image_ids_offset
        offset_cond = offset_cond[1:]
        cond_latent_image_ids = []
        for offset_ in offset_cond:
            cond_latent_image_ids.append(
                self._prepare_image_ids(
                    condition_image_latents.shape[2] // 2,
                    condition_image_latents.shape[3] // 2,
                    offset_w=offset_ * condition_image_latents.shape[3] // 2
                )
            )
    
        if mask_affine is not None:
            affine_H, affine_W = condition_image_latents.shape[2] // 2, condition_image_latents.shape[3] // 2
            scale_factor = 1 / 16
            cond_latent_image_ids_fg = cond_latent_image_ids[0].reshape(affine_H, affine_W, 3).clone()

            # opt 1
            cond_latent_image_ids[0] = warp_affine_tensor(
                cond_latent_image_ids_fg, mask_affine, output_size=(affine_H, affine_W),
                scale_factor=scale_factor, device=self.device,
            )
        cond_latent_image_ids = torch.stack(cond_latent_image_ids)
        
        # Pack condition latents
        cond_image_latents = self._pack_latents(condition_image_latents)
        cond_input = {
            "image_latents": cond_image_latents,
            "image_ids": cond_latent_image_ids,
        }
        # Prepare initial latents
        width, height = bucket_size
        num_channels_latents = self.denoise_model.config.in_channels // 4
        latents, latent_image_ids = self._prepare_latents(
            batch_size, num_channels_latents, height, width, generator
        )

        # Setup timesteps
        sigmas = np.linspace(1.0, 1 / self.config.num_inference_steps, self.config.num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            self.config.num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

        # Denoising loop
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            with torch.autocast(enabled=True, device_type="cuda", dtype=self.config.dtype):
                noise_pred = self.denoise_model(
                    hidden_states=latents,
                    cond_input=cond_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    data_num_per_group=batch_size,
                    image_tags=self.config.image_tags,
                    context_tags=self.config.context_tags,
                    max_sequence_length=self.config.max_sequence_length,
                    mix_attention_double=self.config.mix_attention_double,
                    mix_attention_single=self.config.mix_attention_single,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                if truecfg and i >= 1:
                    guidance_neg = torch.full([1], 1, device=self.device, dtype=torch.float32)
                    guidance_neg = guidance_neg.expand(batch_size)
                    noise_pred_neg = self.denoise_model(
                        hidden_states=latents,
                        cond_input=cond_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        data_num_per_group=batch_size,
                        image_tags=self.config.image_tags,
                        context_tags=self.config.context_tags,
                        max_sequence_length=self.config.max_sequence_length,
                        mix_attention_double=self.config.mix_attention_double,
                        mix_attention_single=self.config.mix_attention_single,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_neg + 5 * (noise_pred - noise_pred_neg)

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode latents
        latents = self._unpack_latents(latents, height, width)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(latents, return_dict=False)[0]
        
        # Post-process images
        images = images.add(1).mul(127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return images

    def _encode_images(self, images):
        return encode_images_cond(self.vae, [images], self.device)

    def _prepare_image_ids(self, h, w, offset_w=0):
        return _prepare_image_ids(h, w, offset_w=offset_w).to(self.device)

    def _pack_latents(self, latents):
        b, c, h, w = latents.shape
        return _pack_latents(latents, b, c, h, w)

    def _unpack_latents(self, latents, height, width):
        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return _unpack_latents(latents, height, width, vae_scale)

    def _prepare_latents(self, batch_size, num_channels_latents, height, width, generator):
        vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        latents, latent_image_ids = prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            vae_downsample_factor=vae_scale,
            height=height,
            width=width,
            dtype=self.config.dtype,
            device=self.device,
            generator=generator,
            offset=None
        )
        return latents, latent_image_ids

def main():
    parser = transformers.HfArgumentParser(InferenceConfig)
    config: InferenceConfig = parser.parse_args_into_dataclasses()[0]
    model = DreamFuseInference(config)
    os.makedirs(config.valid_output_dir, exist_ok=True)
    for valid_root, valid_json in zip(config.valid_roots, config.valid_jsons):
        with open(valid_json, 'r') as f:
            valid_info = json.load(f)
            
        # multi gpu
        to_process = sorted(list(valid_info.keys()))

        sd_idx = len(to_process) // config.total_num * config.sub_idx
        ed_idx = len(to_process) // config.total_num * (config.sub_idx + 1)
        if config.sub_idx < config.total_num - 1:
            print(config.sub_idx, sd_idx, ed_idx)
            to_process = to_process[sd_idx:ed_idx]
        else:
            print(config.sub_idx, sd_idx)
            to_process = to_process[sd_idx:]
        valid_info = {k: valid_info[k] for k in to_process}

        for meta_key, info in tqdm(valid_info.items()):
            img_name = meta_key.split('/')[-1]

            foreground_img = Image.open(os.path.join(valid_root, info['img_info']['000']))
            background_img = Image.open(os.path.join(valid_root, info['img_info']['001']))
            
            new_fg_mask = Image.open(os.path.join(valid_root, info['img_mask_info']['000_mask_scale']))
            ori_fg_mask = Image.open(os.path.join(valid_root, info['img_mask_info']['000']))


            foreground_img.paste((255, 255, 255), mask=ImageOps.invert(ori_fg_mask))

            images = model(foreground_img.copy(), background_img.copy(),
                ori_fg_mask, new_fg_mask,
                prompt=config.ref_prompts,
                seed=config.seed,
                cfg=config.guidance_scale,
                size_select=config.inference_scale,
                text_strength=config.text_strength,
                truecfg=config.truecfg)
            
            result_image = Image.fromarray(images[0], "RGB")
            result_image = result_image.resize(background_img.size)
            result_image.save(os.path.join(config.valid_output_dir, f"{img_name}_2.png"))
            # Make grid
            grid_image = [foreground_img, background_img] + [result_image]
            result = make_image_grid(grid_image, 1, len(grid_image), size=result_image.size)
            
            result.save(os.path.join(config.valid_output_dir, f"{img_name}.jpg"))

if __name__ == "__main__":
    main()
