# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adopt from
# https://github.com/vllm-project/vllm/blob/a0e619e62/vllm/v1/spec_decode/eagle.py
import logging

import torch
import torch.nn as nn

from nanovllm.config import Config
from nanovllm.models.qwen3_eagle3 import Eagle3Qwen3ForCausalLM
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
        # batch_size = next_token_ids.shape[0]
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        input_ids = torch.empty_like(target_token_ids)
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        input_ids[:-1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        input_ids[last_token_indices] = next_token_ids

        logger.debug(
            "eagle draft model forward on:\n  input_ids: %s\n  positions: %s\n  hidden_states: %s",
            input_ids.tolist() if input_ids is not None else None,
            target_positions.tolist() if target_positions is not None else None,
            target_hidden_states.shape,
        )
        hidden_states_for_logits, hidden_states = self.model(
            input_ids=input_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
        )
        sample_hidden_states = hidden_states_for_logits[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        logger.debug(
            "eagle draft model forward get:\n  hidden_states_for_logits: %s\n  hidden_states: %s"
            "\n  draft_token_ids: %s",
            hidden_states_for_logits.shape,
            hidden_states.shape,
            draft_token_ids.tolist(),
        )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        positions = target_positions[last_token_indices]

        hidden_states_fwd = hidden_states[last_token_indices]
        for _ in range(self.num_speculative_tokens - 1):
            # update the inputs
            input_ids = draft_token_ids_list[-1]
            positions += 1

            # update context

            # Run the model.
            hidden_states_for_logits, hidden_states = self.model(
                input_ids=input_ids,
                hidden_states=hidden_states_fwd,
                positions=target_positions,
            )
            logits = self.model.compute_logits(hidden_states_for_logits)
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def load_model(self, target_model: nn.Module) -> None:
        self.model = Eagle3Qwen3ForCausalLM(self.config)
        load_model(self.model, self.config.speculative_config.draft_path)
        self.model.model.embed_tokens = target_model.model.embed_tokens
