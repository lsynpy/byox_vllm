# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adopt from
# https://github.com/vllm-project/vllm/blob/a0e619e62/vllm/v1/spec_decode/eagle.py
import logging

import torch
import torch.nn as nn

from nanovllm.config import Config
from nanovllm.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
from nanovllm.utils.context import get_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__, logging.DEBUG)

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
        # batch_size = next_token_ids.shape[0]
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        input_ids = torch.empty_like(target_token_ids)
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        input_ids[:-1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        input_ids[last_token_indices] = next_token_ids

        # reuse the target context here
        hidden_states_for_logits, hidden_states = self.model(
            input_ids=input_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
        )
        sample_hidden_states = hidden_states_for_logits[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        logger.debug("sampled draft token ids: %s", draft_token_ids.tolist())

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]
        positions = target_positions[last_token_indices]
        hidden_states_fwd = hidden_states[last_token_indices]

        for idx in range(self.num_speculative_tokens - 1):
            input_ids = draft_token_ids_list[-1]
            positions += 1
            self._update_context()
            hidden_states_for_logits, hidden_states_fwd = self.model(
                input_ids=input_ids,
                hidden_states=hidden_states_fwd,
                positions=positions,
            )
            logits = self.model.compute_logits(hidden_states_for_logits)
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)
            logger.debug(
                "draft forward %d. sampled draft_token_ids: %s",
                idx,
                draft_token_ids.tolist(),
            )

        logger.debug("-" * 50)
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        draft_token_ids_list = draft_token_ids.tolist()
        logger.info("draft_token_ids_list: %s", draft_token_ids_list)
        return draft_token_ids_list

    def _update_context(self):
        context = get_context()

        num_sequences = context.cu_seqlens_k.shape[0] - 1

        if num_sequences > 0:
            # Extract the last slot for each sequence and add 1 to get the next slot
            # Do this BEFORE updating cu_seqlens_k since we need the original values
            new_slots_list = []
            for i in range(num_sequences):
                seq_start_idx = context.cu_seqlens_k[i].item()
                seq_end_idx = context.cu_seqlens_k[i + 1].item()
                if seq_start_idx < seq_end_idx and seq_end_idx <= context.slot_mapping.shape[0]:
                    last_slot = context.slot_mapping[seq_end_idx - 1]
                    new_slots_list.append(last_slot + 1)

            # For speculative decoding, we're generating one token per sequence
            cu_seqlens_q = torch.arange(
                num_sequences + 1, dtype=torch.int32, device=context.cu_seqlens_q.device
            )
            cu_seqlens_k = torch.arange(
                num_sequences + 1, dtype=torch.int32, device=context.cu_seqlens_k.device
            )

            max_seqlen_q = 1
            max_seqlen_k = 1

            context.cu_seqlens_q = cu_seqlens_q
            context.cu_seqlens_k = cu_seqlens_k
            context.max_seqlen_q = max_seqlen_q
            context.max_seqlen_k = max_seqlen_k

            if new_slots_list:
                new_slots = torch.tensor(
                    new_slots_list, dtype=torch.int32, device=context.slot_mapping.device
                )
                context.slot_mapping = new_slots
            else:
                new_slots = context.slot_mapping + 1
                context.slot_mapping = new_slots

            if context.context_lens is not None:
                context.context_lens = context.context_lens + 1
            else:
                context.context_lens = torch.ones(
                    num_sequences, dtype=torch.int32, device=context.cu_seqlens_q.device
                )

            if context.block_tables is not None:
                pass
            else:
                num_blocks_needed = num_sequences
                context.block_tables = torch.zeros(
                    (num_blocks_needed, 1), dtype=torch.int32, device=context.cu_seqlens_q.device
                )

    def load_model(self, target_model: nn.Module) -> None:
        self.model = Eagle3Qwen3ForCausalLM(self.config)
        load_model(self.model, self.config.speculative_config.draft_path)
        self.model.model.embed_tokens = target_model.model.embed_tokens
