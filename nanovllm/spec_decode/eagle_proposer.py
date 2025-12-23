# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adopt from
# https://github.com/vllm-project/vllm/blob/a0e619e62/vllm/v1/spec_decode/eagle.py
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import triton
import triton.language as tl

from nanovllm.config import Config
from nanovllm.models.llama_eagle3 import Eagle3Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__, logging.INFO)

PADDING_SLOT_ID = -1


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class EagleProposer:
    def __init__(
        self,
        config: Config,
        device: torch.device,
    ):
        self.config = config
        self.num_speculative_tokens = config.speculative_config.num_speculative_tokens
        self.max_model_len = config.model_config.max_model_len
        self.block_size = config.cache_config.block_size
        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.arange = torch.arange(
            config.scheduler_config.max_num_seqs + 1, device=device, dtype=torch.int32
        )

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [num_tokens]
        target_slot_mapping: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # [batch_size + 1] starting with 0
        cu_num_tokens: torch.Tensor,
        # [batch_size, max_num_blocks_per_req]
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        last_token_indices = cu_num_tokens[1:] - 1

        input_ids = torch.empty_like(target_token_ids)
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        input_ids[:-1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        input_ids[last_token_indices] = next_token_ids

        # FA requires seq_len to have dtype int32.
        seq_lens = (target_positions[last_token_indices] + 1).int()

        # FIXME(woosuk): The below two ops cause synchronization. Optimize.
        max_seq_len = seq_lens.max().item()
        max_num_tokens = (cu_num_tokens[1:] - cu_num_tokens[:-1]).max().item()
        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=max_num_tokens,
            query_start_loc=cu_num_tokens,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=target_slot_mapping,
        )

        hidden_states_logits, hidden_states_fwd = self.model(
            input_ids=input_ids,
            hidden_states=target_hidden_states,
            positions=target_positions,
        )
        sample_hidden_states = hidden_states_logits[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        positions = target_positions[last_token_indices]
        hidden_states = hidden_states_fwd[last_token_indices]
        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        for _ in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            input_ids = draft_token_ids_list[-1]
            positions += 1

            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            exceeds_max_model_len = positions >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)

            # Increment the sequence lengths.
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len, self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = block_table.gather(dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (
                block_ids * self.block_size + clamped_positions % self.block_size
            )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

            # Run the model.
            hidden_states_logits, hidden_states = self.model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                positions=clamped_positions,
            )
            logits = self.model.compute_logits(hidden_states_logits, None)
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    @staticmethod
    def prepare_inputs(
        # [batch_size + 1]
        cu_target_query_lens: torch.Tensor,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = cu_target_query_lens[1:] - cu_target_query_lens[:-1]
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens

        cu_num_tokens = torch.empty_like(cu_target_query_lens)
        torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
        cu_num_tokens[0] = 0

        # FIXME(woosuk): Avoid synchronization.
        num_tokens = cu_num_tokens[-1].item()
        token_indices = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=cu_num_tokens.device,
        )

        batch_size = num_rejected_tokens.shape[0]
        BLOCK_SIZE = 1024
        prepare_input_kernel[(batch_size,)](
            token_indices,
            cu_target_query_lens,
            cu_num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return cu_num_tokens, token_indices

    def load_model(self, target_model: nn.Module) -> None:
        target_layer_num = self.config.model_config.get_num_layers(self.config.parallel_config)
        draft_model_config = self.config.speculative_config.draft_model_config
        target_device = self.config.device_config.device
        self.model = Eagle3Qwen3ForCausalLM(
            model_config=draft_model_config, start_layer_id=target_layer_num
        ).to(target_device)

        load_model(self.model, self.config.speculative_config.draft_path)
        self.model.model.embed_tokens = target_model.model.embed_tokens


@triton.jit
def prepare_input_kernel(
    out_ptr,
    cu_query_lens_ptr,
    cu_num_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # [start_pos, end_pos)
    start_pos = tl.load(cu_num_tokens_ptr + pid)
    end_pos = tl.load(cu_num_tokens_ptr + pid + 1)
    num_tokens = end_pos - start_pos

    index_start = tl.load(cu_query_lens_ptr + pid)

    num_blocks = tl.cdiv(num_tokens, BLOCK_SIZE)
    for i in tl.range(num_blocks):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            out_ptr + start_pos + offset,
            index_start + offset,
            mask=offset < num_tokens,
        )
