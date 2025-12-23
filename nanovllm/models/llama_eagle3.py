# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adopt from
# https://github.com/vllm-project/vllm/blob/a0e619e62/vllm/model_executor/models/llama_eagle3.py

import logging

import torch
import torch.nn as nn
from transformers import Qwen3Config

from nanovllm.config import Config
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.models.qwen3 import QKVParallelLinear, Qwen3DecoderLayer, Qwen3ForCausalLM
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__, logging.INFO)


class Eagle3Qwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)

        # override qkv - note: using config.hidden_size instead of self.hidden_size
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * config.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Eagle3Qwen3Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config.speculative_config.draft_hf_config
        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [Eagle3Qwen3DecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        if hasattr(self.config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                self.config.target_hidden_size * 3, self.config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(self.config.hidden_size * 3, self.config.hidden_size, bias=False)
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        assert hidden_states.shape[-1] == input_embeds.shape[-1]

        residual = None
        hidden_states, residual = self.layers[0](
            positions,
            input_embeds,
            hidden_states,
            residual,
        )

        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm


class Eagle3Qwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: Config):
        nn.Module.__init__(self)
        self.config = config.speculative_config.draft_hf_config
        self.model = Eagle3Qwen3Model(config=config)

        # Use vocab_size from the draft config
        draft_vocab_size = getattr(self.config, "vocab_size", self.config.vocab_size)
        self.lm_head = ParallelLMHead(
            draft_vocab_size,
            self.config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # Use greedy sampling - return raw logits from lm_head
        return self.lm_head(hidden_states)

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # combine multiple auxiliary hidden states returned by eagle3
        return self.model.fc(hidden_states)
