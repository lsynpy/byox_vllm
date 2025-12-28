# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adopt from
# https://github.com/vllm-project/vllm/blob/a0e619e62/vllm/v1/spec_decode/eagle.py
import logging

import torch
import torch.nn as nn
import triton
import triton.language as tl

from nanovllm.config import Config
from nanovllm.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
from nanovllm.utils.context import get_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__, logging.INFO)

PADDING_SLOT_ID = -1


class EagleProposer:
    def __init__(
        self,
        config: Config,
        device: torch.device,
    ):
        self.config = config
        self.num_speculative_tokens = config.speculative_config.num_speculative_tokens
        self.max_model_len = config.max_model_len
        self.block_size = config.kvcache_block_size
        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.arange = torch.arange(config.max_num_seqs + 1, device=device, dtype=torch.int32)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        logger.debug(
            "propose inputs: \n  target_token_ids: %s \n  target_positions: %s"
            "\n  target_hidden_states: %s \n  next_token_ids: %s \n last_token_indices: %s",
            target_token_ids.tolist(),
            target_positions.tolist(),
            target_hidden_states.shape,
            next_token_ids.tolist(),
            last_token_indices.tolist(),
        )
        batch_size = next_token_ids.shape[0]
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        input_ids = torch.empty_like(target_token_ids)
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        input_ids[:-1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        input_ids[last_token_indices] = next_token_ids

        # FA requires seq_len to have dtype int32.
        seq_lens = (target_positions[last_token_indices] + 1).int()

        max_seq_len = seq_lens.max().item()
        context = get_context()
        context.max_seq_len = max_seq_len
        context.context_lens = None

        hidden_states_logits, hidden_states_fwd = self.model(
            input_ids=input_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
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
        context.num_actual_tokens = batch_size
        context.max_query_len = 1
        context.query_start_loc = self.arange[: batch_size + 1]
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
            context.max_seq_len += 1
            context.seq_lens += 1
            # Consider max model length.
            context.max_seq_len = min(context.max_seq_len, self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            context.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = context.block_table.gather(dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            context.slot_mapping = block_ids * self.block_size + clamped_positions % self.block_size
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            context.slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

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
        self.model = Eagle3Qwen3ForCausalLM(self.config)
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
