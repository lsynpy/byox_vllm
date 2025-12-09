# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

INVALID_TOKEN_ID = -1


class RejectionSampler(nn.Module):
    def forward(self, logits: torch.Tensor, spec_token_ids: list[list[int]]) -> list[int]:
        return RejectionSampler.greedy_sample_native(logits, spec_token_ids)

    @staticmethod
    def greedy_sample_native(logits: torch.Tensor, spec_token_ids: list[list[int]]) -> list[int]:
        spec_lens: list[int] = [len(x) for x in spec_token_ids]
        # Add 1 to include the 'bonus' token.
        sample_lens: list[int] = [x + 1 for x in spec_lens]

        output_token_ids: torch.Tensor = logits.argmax(dim=-1).view(-1)
        output_token_ids = output_token_ids.split(sample_lens)
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=INVALID_TOKEN_ID)

        # Convert spec token IDs to a tensor, split by sample_lens, then pad.
        spec_token_ids: list[torch.Tensor] = [
            torch.tensor(x, dtype=output_token_ids.dtype, device=output_token_ids.device)
            for x in spec_token_ids
        ]
        spec_token_ids = pad_sequence(spec_token_ids, batch_first=True, padding_value=INVALID_TOKEN_ID)

        # Produce a mask that remains 1 (True) until the first
        # mismatch (cumprod turns 0 after a mismatch).
        accept_mask = (output_token_ids[:, :-1] == spec_token_ids).cumprod(dim=1)
        # Identify valid positions (non-padding).
        valid_mask = output_token_ids != INVALID_TOKEN_ID
        # Generate mask with bonus token.
        generate_mask = (
            torch.cat(
                [accept_mask, torch.zeros(accept_mask.size(0), 1, device=accept_mask.device)], dim=1
            ).to(torch.bool)
            & valid_mask
        )
        zeros_mask = generate_mask == 0
        first_zero_idx = zeros_mask.float().argmax(dim=1)
        # Figure out which rows actually contain at least one zero.
        rows_with_zero = zeros_mask.any(dim=1)
        # Use indexing to set the first zero in each of those rows to 1.
        generate_mask[rows_with_zero, first_zero_idx[rows_with_zero]] = 1

        output_token_ids[~generate_mask] = INVALID_TOKEN_ID

        output_token_ids = output_token_ids.tolist()
        output_token_ids = [[token_id for token_id in lst if token_id != -1] for lst in output_token_ids]
        return output_token_ids
