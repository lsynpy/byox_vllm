import torch
from torch import nn

from nanovllm.sample.metadata import SamplingMetadata


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def _greedy_sample(self, logits: torch.Tensor):
        return torch.argmax(logits, dim=-1)

    def _stochastic_sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits_scaled = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits_scaled, dim=-1)
        sampled_probs = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10))
        return sampled_probs.argmax(dim=-1)

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        sampling_metadata: SamplingMetadata = None,
        predict_bonus_token: bool = False,
    ):
        temp_zero_mask = temperatures == 0

        sample_tokens = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        if torch.any(temp_zero_mask):
            greedy_logits = logits[temp_zero_mask]
            greedy_tokens = self._greedy_sample(greedy_logits)
            sample_tokens[temp_zero_mask] = greedy_tokens

        non_zero_temp_mask = ~temp_zero_mask
        if torch.any(non_zero_temp_mask):
            relevant_logits = logits[non_zero_temp_mask]
            relevant_temps = temperatures[non_zero_temp_mask]
            normal_tokens = self._stochastic_sample(relevant_logits, relevant_temps)
            sample_tokens[non_zero_temp_mask] = normal_tokens

        return sample_tokens
