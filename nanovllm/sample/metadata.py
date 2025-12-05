# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    generators: dict[int, torch.Generator]
    prompt_token_ids: torch.Tensor | None
    output_token_ids: list[list[int]]
    spec_token_ids: list[list[int]] | None = None
