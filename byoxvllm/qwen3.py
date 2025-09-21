import torch
from einops import rearrange
from torch import nn
from transformers import Qwen3Config

from byoxvllm.layers import Attention, RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads

        attn_bias = config.attention_bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
        )

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_kv_heads, d=self.head_dim)
        q = rearrange(q, "b l (h d) -> (b l) h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b l (h d) -> (b l) h d", h=self.num_kv_heads, d=self.head_dim)
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rearrange(q, "(b l) h d -> b h l d", b=bsz, l=q_len)
        k = rearrange(k, "(b l) h d -> b h l d", b=bsz, l=q_len)

        print(f"q -> {q.flatten()[-3:]}, {q.shape}")
        print(f"k -> {k.flatten()[-3:]}, {k.shape}")
        print(f"v -> {v.flatten()[-3:]}, {v.shape}")

        cos, sin = self.rotary_emb(q.dtype, positions)
        old_q = q
        old_k = k
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        print(f"{torch.sum(old_k != k)} different values in total {k.numel()}")
        print(f"{torch.sum(old_q != q)} different values in total {q.numel()}")

        print(f"q -> {q.flatten()[-3:]}, {q.shape}")
        print(f"k -> {k.flatten()[-3:]}, {k.shape}")
        print(f"v -> {v.flatten()[-3:]}, {v.shape}")

        attn_output = self.attn(q, k, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        act_output = self.act_fn(gate_output)
        x = act_output * up_output
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(positions, hidden_states)
            break

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return self.lm_head(hidden_states)
