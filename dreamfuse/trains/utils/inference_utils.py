import torch
from diffusers.utils.torch_utils import randn_tensor
import numpy as np
from einops import rearrange
import torch.nn.functional as F

def get_mask_affine(mask1, mask2):
    box1 = mask1.getbbox()
    box2 = mask2.getbbox()

    if box1 is None or box2 is None:
        affine_coeffs = [1, 0, 0, 0, 1, 0]
        return affine_coeffs

    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    w1, h1 = right1 - left1, bottom1 - top1
    w2, h2 = right2 - left2, bottom2 - top2

    scale_x = w1 / w2
    scale_y = h1 / h2

    tx = left1 - left2*scale_x
    ty = top1 - top2*scale_y

    affine_coeffs = [scale_x, 0, tx, 0, scale_y, ty]
    return affine_coeffs

def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def compute_text_embeddings(config, prompt, text_encoders, tokenizers, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, config.max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def _prepare_image_ids(height, width, offset_h=0, offset_w=0):
    image_ids = torch.zeros(height, width, 3)
    image_ids[..., 1] = image_ids[..., 1] + torch.arange(height)[:, None] + offset_h
    image_ids[..., 2] = image_ids[..., 2] + torch.arange(width)[None, :] + offset_w
    image_ids = image_ids.reshape(-1, 3)
    return image_ids


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents

def _unpack_latents(latents, height, width, vae_downsample_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_downsample_factor * 2))
    width = 2 * (int(width) // (vae_downsample_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype, offset_h=0, offset_w=0):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None] + offset_h
    )
    latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :] + offset_w
    )

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
        latent_image_ids.shape
    )

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def pil_to_tensor(image, device="cpu"):
    image = np.array(image)
    image = torch.from_numpy(image).float() / 127.5 - 1.0
    image = image.permute(2, 0, 1).to(device)
    return image

@torch.no_grad()
def encode_images_cond(vae_model, condition_images, device):
    condition_image_tensors = []
    for condition_image in condition_images:
        condition_image_tensor = torch.tensor(np.array(condition_image)).to(device).permute(0, 3, 1, 2) # shape: [n_cond, c, h, w]
        condition_image_tensor = condition_image_tensor / 127.5 - 1.0
        condition_image_tensors.append(condition_image_tensor)
    condition_image_tensors = torch.stack(condition_image_tensors) # shape: [bs, n_cond, c, h, w]
    condition_image_tensors = rearrange(condition_image_tensors, 'b n c h w -> (b n) c h w')

    # encode condition images
    condition_image_latents = (
        vae_model.encode(
            condition_image_tensors.to(vae_model.dtype)
        ).latent_dist.sample()
    ) # shape: [bs*n_cond, c, h // 8, w // 8]
    condition_image_latents = (condition_image_latents - vae_model.config.shift_factor) * vae_model.config.scaling_factor

    return condition_image_latents


def prepare_latents(
        batch_size,
        num_channels_latents,
        vae_downsample_factor,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        offset=None,
        hw=False,
):
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_downsample_factor * 2))
    width = 2 * (int(width) // (vae_downsample_factor * 2))

    shape = (batch_size, num_channels_latents, height, width)

    if latents is not None:
        if offset is None:
            latent_image_ids = _prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
        else:
            latent_image_ids = []
            for offset_ in offset:
                latent_image_ids.append(
                    _prepare_latent_image_ids(
                        batch_size, height // 2, width // 2, device, dtype, offset_w=offset_ * width // 2, offset_h=offset_ * height // 2 if hw else 0
                    )
                )
        return latents.to(device=device, dtype=dtype), latent_image_ids

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    latents = _pack_latents(
        latents, batch_size, num_channels_latents, height, width
    )
    if offset is None:
        latent_image_ids = _prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )
    else:
        latent_image_ids = []
        for offset_ in offset:
            latent_image_ids.append(
                _prepare_latent_image_ids(
                    batch_size, height // 2, width // 2, device, dtype, offset_w=offset_ * width // 2,  offset_h=offset_ * height // 2 if hw else 0
                )
            )
    return latents, latent_image_ids


@torch.no_grad()
def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def warp_affine_tensor(input_tensor, mask_affines, output_size, scale_factor=1/16,
                       align_corners_grid=False, align_corners_sample=True, 
                       flatten_output=True, device=None):
    """
    对输入的 tensor 应用 affine 仿射变换，并返回 warp 后的结果。

    参数：
      input_tensor: 待变换的图像 tensor，支持的形状包括 (H, W, C)、(C, H, W) 或 (1, C, H, W)。
      mask_affines: 仿射参数（例如 [a, 0, tₓ, 0, e, t_y]），这些参数单位基于 512×512 图像。
      output_size: 目标输出的空间尺寸，格式为 (H_out, W_out)。
      scale_factor: 平移参数的缩放因子；例如若 512→32，则 factor = 32/512 = 1/16。
      align_corners_grid: 传递给 F.affine_grid 的 align_corners 参数。
      align_corners_sample: 传递给 F.grid_sample 的 align_corners 参数。
      flatten_output: 若为 True，则将输出 warp 后的 tensor 从 (1, C, H_out, W_out) 转换为 (-1, C)。
      device: 如果设置，将将相关 tensor 移动到指定的设备上。

    返回：
      warped_output: 经过 affine warp 处理后的 tensor，
                      若 flatten_output 为 True，则形状为 (H_out*W_out, C)，否则为 (1, C, H_out, W_out)。
    """
    # 如果输入 tensor 不是 batch（4D）的，则调整为 (1, C, H, W)
    if input_tensor.dim() == 3:
        # 判断是否为 (H, W, C)，如果最后一维为 3，则认为是 RGB
        if input_tensor.shape[-1] == 3:
            input_tensor = input_tensor.permute(2, 0, 1)
        input_tensor = input_tensor.unsqueeze(0)
    elif input_tensor.dim() != 4:
        raise ValueError("input_tensor 必须是 3D 或 4D Tensor！")
    
    # 输出尺寸
    H_out, W_out = output_size
    B, C, H_in, W_in = input_tensor.shape

    # 将 mask_affines 转换为 tensor，确保形状为 (1, 6)
    if not torch.is_tensor(mask_affines):
        theta = torch.tensor(mask_affines, dtype=torch.float32).unsqueeze(0)
    else:
        theta = mask_affines.clone().float()
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
    
    # 调整平移部分（第三和第六个元素），使其适应当前目标分辨率
    theta[0, 2] *= scale_factor  # x 方向平移
    theta[0, 5] *= scale_factor  # y 方向平移

    a   = theta[0, 0]
    t_x = theta[0, 2]
    e   = theta[0, 4]
    t_y = theta[0, 5]
    
    # 根据归一化转换（范围 [-1, 1]）
    # 对 x 方向：归一化公式为 x_norm = 2*x/(W_out-1) - 1
    # 转换后 affine 的常数项即为：a + 2*t_x/(W_out-1) - 1
    theta_norm = torch.tensor([
        [a, 0.0, a + 2*t_x/(W_out - 1) - 1],
        [0.0, e, e + 2*t_y/(H_out - 1) - 1]
    ], dtype=torch.float32).unsqueeze(0)

    # 根据目标输出大小创建 affine_grid，grid 的 size 为 (B, C, H_out, W_out)
    grid = F.affine_grid(theta_norm, size=(B, C, H_out, W_out), align_corners=align_corners_grid)
    if device is not None:
        grid = grid.to(device)
        input_tensor = input_tensor.to(device)

    # 对输入 tensor 进行采样
    warped = F.grid_sample(input_tensor, grid, align_corners=align_corners_sample)
    
    # 若需要将输出展平为 (-1, C)
    if flatten_output:
        # 将 (1, C, H_out, W_out) → 转为 (H_out, W_out, C) → reshape(-1, C)
        warped = warped.squeeze(0).permute(1, 2, 0).reshape(-1, C)
    return warped

def find_nearest_bucket_size(input_width, input_height, mode="x64", bucket_size=512):
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


