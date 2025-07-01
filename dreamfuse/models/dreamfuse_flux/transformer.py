# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from dreamfuse.models.dreamfuse_flux.flux_processor import FluxAttnSharedProcessor2_0
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .flux_processor import CombinedTimestepGuidanceTextProjEmbeddings

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, cross_attention_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_head = cross_attention_dim // heads
        self.attn_to_q = nn.Linear(query_dim, cross_attention_dim, bias=bias)
        self.norm_q = nn.LayerNorm(self.dim_head)

        self.attn_to_k = nn.Linear(cross_attention_dim, cross_attention_dim, bias=bias)
        self.norm_k = nn.LayerNorm(self.dim_head)

        self.attn_to_v = nn.Linear(cross_attention_dim, cross_attention_dim, bias=bias)

        self.attn_to_out = nn.ModuleList([])
        self.attn_to_out.append(nn.Linear(query_dim, query_dim, bias=bias))
        self.attn_to_out.append(nn.Dropout(dropout))
        
        # zero init
        with torch.no_grad():
            self.attn_to_out[0].weight.fill_(0)
            # self.to_out[0].bias.fill_(0)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.attn_to_q(hidden_states)
        key = self.attn_to_k(encoder_hidden_states)
        value = self.attn_to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

        hidden_states = self.attn_to_out[0](hidden_states)
        hidden_states = self.attn_to_out[1](hidden_states)

        return hidden_states

@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnSharedProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            data_num_per_group=1,
            max_sequence_length=512,
            mix_attention: bool = True,
            cond_temb = None,
            cond_image_rotary_emb = None,
            cond_latents = None,
            joint_attention_kwargs=None,

    ):
        with_cond = cond_latents is not None and mix_attention
        
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states)) 

        if with_cond:
            residual_cond = cond_latents
            norm_cond_latents, cond_gate = self.norm(cond_latents, emb=cond_temb)
            mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_cond_latents))
        
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            data_num_per_group=data_num_per_group,
            max_sequence_length=max_sequence_length,
            mix_attention=mix_attention,
            cond_latents=norm_cond_latents if with_cond else None,
            cond_image_rotary_emb=cond_image_rotary_emb if with_cond else None,
            **joint_attention_kwargs,
        )
        
        if with_cond:
            attn_output, cond_attn_output = attn_output

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        
        if with_cond:
            cond_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
            cond_gate = cond_gate.unsqueeze(1)
            cond_latents = cond_gate * self.proj_out(cond_latents)
            cond_latents = residual_cond + cond_latents
        
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if with_cond:
            return hidden_states, cond_latents
        else:
            return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        processor = FluxAttnSharedProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            data_num_per_group=1,
            max_sequence_length=512,
            mix_attention: bool = True,
            cond_temb = None,
            cond_image_rotary_emb = None,
            cond_latents = None,
            joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        with_cond = cond_latents is not None and mix_attention
        if with_cond:
            norm_cond_latents, cond_gate_msa, cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = self.norm1(cond_latents, emb=cond_temb)

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            data_num_per_group=data_num_per_group,
            max_sequence_length=max_sequence_length,
            mix_attention=mix_attention,
            cond_latents=norm_cond_latents if with_cond else None,
            cond_image_rotary_emb=cond_image_rotary_emb if with_cond else None,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3 and with_cond:
            attn_output, context_attn_output, cond_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3 and not with_cond:
            hidden_states = hidden_states + ip_attn_output

        if with_cond:
            cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
            cond_latents = cond_latents + cond_attn_output
            
            norm_cond_latents = self.norm2(cond_latents)
            norm_cond_latents = norm_cond_latents * (1 + cond_scale_mlp[:, None]) + cond_shift_mlp[:, None]

            cond_ff_output = self.ff(norm_cond_latents)
            cond_ff_output = cond_gate_mlp.unsqueeze(1) * cond_ff_output

            cond_latents = cond_latents + cond_ff_output
        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        if with_cond:
            return encoder_hidden_states, hidden_states, cond_latents
        else:
            return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, FluxTransformer2DLoadersMixin
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
            self,
            patch_size: int = 1,
            in_channels: int = 64,
            out_channels: Optional[int] = None,
            num_layers: int = 19,
            num_single_layers: int = 38,
            attention_head_dim: int = 128,
            num_attention_heads: int = 24,
            joint_attention_dim: int = 4096,
            pooled_projection_dim: int = 768,
            guidance_embeds: bool = False,
            axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        if getattr(self.config, "num_image_tag_embeddings", None) is not None:
            self.image_tag_embeddings = nn.Embedding(self.config.num_image_tag_embeddings, self.inner_dim)
        if getattr(self.config, "num_context_tag_embeddings", None) is not None:
            self.context_tag_embeddings = nn.Embedding(self.config.num_context_tag_embeddings, self.inner_dim)

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def set_tag_embeddings(self, num_image_tag_embeddings=0, num_context_tag_embeddings=0):
        if num_image_tag_embeddings > 0:
            self.config.num_image_tag_embeddings = num_image_tag_embeddings
            self.image_tag_embeddings = zero_module(nn.Embedding(self.config.num_image_tag_embeddings, self.inner_dim))
        if num_context_tag_embeddings > 0:
            self.config.num_context_tag_embeddings = num_context_tag_embeddings
            self.context_tag_embeddings = zero_module(nn.Embedding(self.config.num_context_tag_embeddings, self.inner_dim))

    def set_mask_tokenizer(self, mask_in_chans, mask_out_chans, activation = nn.GELU):
        self.mask_tokenizer = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, padding=1),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, mask_out_chans, kernel_size=1),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.mask_attn = CrossAttention(mask_out_chans, mask_out_chans)

    def forward_mask_attn(self, mask_images, fg_images):
        mask_images = self.mask_tokenizer(mask_images)
        mask_images = mask_images.flatten(2).transpose(1, 2)
        mask_images = self.mask_attn(mask_images, fg_images, attention_mask=None)
        return mask_images

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _format_input(self):
        pass

    def _format_output(self):
        pass

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            cond_input: dict = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
            data_num_per_group: int = 1,
            image_tags=None,
            context_tags=None,
            max_sequence_length: int = 512,
            mix_attention_double=True,
            mix_attention_single=True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        mask_cond = None
        mask_ids = None
        if cond_input is not None:
            cond_image_latents = cond_input["image_latents"]
            cond_image_ids = cond_input["image_ids"]
            cond_latents = self.x_embedder(cond_image_latents)

            if joint_attention_kwargs is not None and "mask_cond" in joint_attention_kwargs:
                mask_cond = joint_attention_kwargs.pop("mask_cond")
                mask_ids = joint_attention_kwargs.pop("mask_ids")
                if mask_cond is not None:
                    mask_cond = self.forward_mask_attn(mask_cond, cond_latents[:1])
                # joint_attention_kwargs["mask_cond"] = mask_cond
                # hidden_states = hidden_states + mask_cond

        if image_tags is not None:
            image_tag_embeddings = self.image_tag_embeddings(
                torch.Tensor(
                    image_tags,
                ).to(device=hidden_states.device, dtype=torch.int64)
            )
            bsz = hidden_states.shape[0] // data_num_per_group
            image_tag_embeddings = image_tag_embeddings.repeat_interleave(bsz, dim=0)
            if cond_input is not None:
                hidden_states = hidden_states + image_tag_embeddings[0]
                cond_latents = cond_latents + image_tag_embeddings[1:].unsqueeze(1)
            else:
                # for debug
                if len(hidden_states) != len(image_tag_embeddings):
                    hidden_states += image_tag_embeddings[:1].unsqueeze(1)
                else:
                    hidden_states = hidden_states + image_tag_embeddings.unsqueeze(1)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        if cond_input is not None:
            cond_time = 0
            cond_temb = ( self.time_text_embed(torch.ones_like(timestep)*cond_time, pooled_projections)
                if guidance is None
                else self.time_text_embed(torch.ones_like(timestep)*cond_time, guidance, pooled_projections)
            )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if context_tags is not None:
            context_tag_embeddings = self.context_tag_embeddings(
                torch.Tensor(
                    image_tags,
                ).to(device=hidden_states.device, dtype=torch.int64)
            )
            bsz = hidden_states.shape[0] // data_num_per_group
            context_tag_embeddings = context_tag_embeddings.repeat_interleave(bsz, dim=0)
            if cond_input is not None:
                encoder_hidden_states = encoder_hidden_states + context_tag_embeddings[0]
            else:
                if len(encoder_hidden_states) != len(context_tag_embeddings):
                    encoder_hidden_states += context_tag_embeddings[:1].unsqueeze(1)
                else:
                    encoder_hidden_states = encoder_hidden_states + context_tag_embeddings.unsqueeze(1)

        if mask_cond is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states, mask_cond], dim=1) # todo: compare with add
            max_sequence_length = encoder_hidden_states.shape[1]

            txt_ids = torch.cat((txt_ids, mask_ids), dim=0)

        if isinstance(img_ids, list):
            image_rotary_emb = []
            for img_ids_ in img_ids:
                ids = torch.cat((txt_ids, img_ids_), dim=0)
                image_rotary_emb.append(self.pos_embed(ids))
            image_rotary_emb = (  # to batch, cos / sin
                torch.stack([_[0] for _ in image_rotary_emb]).repeat_interleave(hidden_states.shape[0] // len(img_ids), dim=0).clone(),
                torch.stack([_[1] for _ in image_rotary_emb]).repeat_interleave(hidden_states.shape[0] // len(img_ids), dim=0).clone(),
            )
        else:
            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.pos_embed(ids)
            if cond_input is not None:
                cond_rotary_emb = []
                for image_ids in cond_image_ids:
                    cond_rotary_emb.append(self.pos_embed(image_ids))
                cond_rotary_emb = (
                    torch.stack([_[0] for _ in cond_rotary_emb]).repeat_interleave(cond_latents.shape[0] // len(cond_image_ids), dim=0).clone(),
                    torch.stack([_[1] for _ in cond_rotary_emb]).repeat_interleave(cond_latents.shape[0] // len(cond_image_ids), dim=0).clone(),
                )

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # ckpt_kwargs.updata(joint_attention_kwargs)
                block_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    data_num_per_group,
                    max_sequence_length,
                    mix_attention_double,
                    cond_temb if cond_input is not None else None,
                    cond_rotary_emb if cond_input is not None else None,
                    cond_latents if cond_input is not None else None,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                block_output = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    data_num_per_group=data_num_per_group,
                    max_sequence_length=max_sequence_length,
                    mix_attention=mix_attention_double,
                    cond_temb = cond_temb if cond_input is not None else None,
                    cond_image_rotary_emb = cond_rotary_emb if cond_input is not None else None,
                    cond_latents = cond_latents if cond_input is not None else None,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if cond_input is not None and mix_attention_double:
                encoder_hidden_states, hidden_states, cond_latents = block_output
            else:
                encoder_hidden_states, hidden_states = block_output
                
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                    
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    data_num_per_group,
                    max_sequence_length,
                    mix_attention_single,
                    cond_temb if cond_input is not None else None,
                    cond_rotary_emb if cond_input is not None else None,
                    cond_latents if cond_input is not None else None,
                    joint_attention_kwargs,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    data_num_per_group=data_num_per_group,
                    max_sequence_length=max_sequence_length,
                    mix_attention=mix_attention_single,
                    cond_temb = cond_temb if cond_input is not None else None,
                    cond_image_rotary_emb = cond_rotary_emb if cond_input is not None else None,
                    cond_latents = cond_latents if cond_input is not None else None,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            
            if cond_input is not None and mix_attention_single:
                hidden_states, cond_latents = hidden_states

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
