import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_torch_version, maybe_allow_in_graph
from diffusers.models.attention import Attention
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, PixArtAlphaTextProjection

class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        if (guidance >= 0).all():
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))  # (N, D)

            time_guidance_emb = timesteps_emb + guidance_emb

            pooled_projections = self.text_embedder(pooled_projection)
            conditioning = time_guidance_emb + pooled_projections
        else:
            pooled_projections = self.text_embedder(pooled_projection)
            conditioning = timesteps_emb + pooled_projections

        return conditioning


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if cos.ndim == 2:
            cos = cos[None, None]
        else:
            cos = cos.unsqueeze(1)
        if sin.ndim == 2:
            sin = sin[None, None]
        else:
            sin = sin.unsqueeze(1)
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)

class FluxAttnSharedProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            data_num_per_group: Optional[int] = 1,
            max_sequence_length: Optional[int] = 512,
            mix_attention: bool = True,
            cond_latents = None,
            cond_image_rotary_emb = None,
            work_mode = None, 
            mask_cond = None,
    ) -> torch.FloatTensor:
        with_cond = cond_latents is not None and mix_attention
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)


        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if with_cond:
            cond_bs = cond_latents.shape[0]

            # update condition
            cond_query = attn.to_q(cond_latents)
            cond_query = cond_query.view(cond_bs, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None:
                cond_query = attn.norm_q(cond_query)
            cond_query = apply_rotary_emb(cond_query, cond_image_rotary_emb)
            cond_query = torch.cat(cond_query.chunk(len(cond_query), dim=0), dim=2)
            
            cond_key = attn.to_k(cond_latents)
            cond_value = attn.to_v(cond_latents)
            cond_key = cond_key.view(cond_bs, -1, attn.heads, head_dim).transpose(1, 2)
            cond_value = cond_value.view(cond_bs, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_k is not None:
                cond_key = attn.norm_k(cond_key)
            
            cond_key = apply_rotary_emb(cond_key, cond_image_rotary_emb)

            cond_key = torch.cat(cond_key.chunk(len(cond_key), dim=0), dim=2)
            cond_value = torch.cat(cond_value.chunk(len(cond_value), dim=0), dim=2)

        if data_num_per_group > 1 and mix_attention:
            E = max_sequence_length  # according to text len

            key_enc, key_hid = key[:, :, :E], key[:, :, E:]
            value_enc, value_hid = value[:, :, :E], value[:, :, E:]

            key_layer = key_hid.chunk(data_num_per_group, dim=0)
            key_layer = torch.cat(key_layer, dim=2).repeat(data_num_per_group, 1, 1, 1)

            value_layer = value_hid.chunk(data_num_per_group, dim=0)
            value_layer = torch.cat(value_layer, dim=2).repeat(data_num_per_group, 1, 1, 1)

            key = torch.cat([key_enc, key_layer], dim=2)
            value = torch.cat([value_enc, value_layer], dim=2)
            
        elif data_num_per_group == 1 and mix_attention and with_cond:
            E = max_sequence_length  # according to text len

            key_enc, key_hid = key[:, :, :E], key[:, :, E:]
            value_enc, value_hid = value[:, :, :E], value[:, :, E:]

            # todo: support bs != 1
            key_layer = torch.cat([key_hid, cond_key], dim=2)
            value_layer = torch.cat([value_hid, cond_value], dim=2)

            key = torch.cat([key_enc, key_layer], dim=2)
            value = torch.cat([value_enc, value_layer], dim=2)
            
            # concat query
            query_enc, query_hid = query[:, :, :E], query[:, :, E:]
            query_layer = torch.cat([query_hid, cond_query], dim=2)
            query = torch.cat([query_enc, query_layer], dim=2)
            
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            if with_cond:
                encoder_hidden_states, hidden_states, cond_latents = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] : -cond_latents.shape[1]*cond_bs],
                    hidden_states[:, -cond_latents.shape[1]*cond_bs :],
                )
                cond_latents = cond_latents.view(cond_bs, cond_latents.shape[1] // cond_bs, cond_latents.shape[2])
                cond_latents = attn.to_out[0](cond_latents)
                cond_latents = attn.to_out[1](cond_latents)
            else:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1]:],
                )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            if with_cond:
                return hidden_states, encoder_hidden_states, cond_latents
            return hidden_states, encoder_hidden_states
        else:
            if with_cond:
                hidden_states, cond_latents = (
                    hidden_states[:, : -cond_latents.shape[1]*cond_bs],
                    hidden_states[:, -cond_latents.shape[1]*cond_bs :],
                )
                cond_latents = cond_latents.view(cond_bs, cond_latents.shape[1] // cond_bs, cond_latents.shape[2])
                return hidden_states, cond_latents
            return hidden_states
