# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from nanovllm.sample.rejection_sampler import RejectionSampler


@pytest.fixture
def sampler():
    return RejectionSampler()


def create_logits_tensor(token_ids: list[int], vocab_size: int = 100) -> torch.Tensor:
    """Helper function to create logits tensor that
    will produce desired token ids on argmax"""
    logits = torch.full((len(token_ids), vocab_size), -100.0).cuda()
    for i, token_id in enumerate(token_ids):
        logits[i, token_id] = 100.0
    return logits


def test_perfect_match(sampler):
    """Test when output tokens perfectly match speculated tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [1, 2, 3, 4]  # 4 is the bonus token

    logits = create_logits_tensor(output_tokens)
    output = sampler(logits, spec_tokens)
    expected = [[1, 2, 3, 4]]  # Now returns a list
    assert output == expected


def test_early_mismatch(sampler):
    """Test when there's an early mismatch in tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [1, 5, 3, 4]  # Mismatch at position 1

    logits = create_logits_tensor(output_tokens)
    output = sampler(logits, spec_tokens)
    expected = [[1, 5]]  # Now returns a list - only valid tokens
    assert output == expected


def test_multiple_sequences(sampler):
    """Test handling multiple sequences of speculated tokens"""
    spec_tokens = [[1, 2], [3]]
    output_tokens = [1, 2, 5, 3, 4]  # Two sequences with bonus tokens 5 and 4

    logits = create_logits_tensor(output_tokens)
    output = sampler(logits, spec_tokens)
    expected = [[1, 2, 5], [3, 4]]  # Now returns a list - only valid tokens
    assert output == expected


def test_single_token_sequence(sampler):
    """Test handling sequences with single token"""
    spec_tokens = [[1]]
    output_tokens = [1, 2]  # Single token with bonus token 2

    logits = create_logits_tensor(output_tokens)
    output = sampler(logits, spec_tokens)
    expected = [[1, 2]]  # Now returns a list
    assert output == expected


def test_empty_sequence(sampler):
    """Test handling empty sequence of speculated tokens"""
    spec_tokens: list[list[int]] = [[]]
    output_tokens = [5]  # Just the bonus token

    logits = create_logits_tensor(output_tokens)
    output = sampler(logits, spec_tokens)
    expected = [[5]]  # Now returns a list
    assert output == expected


def test_multiple_mismatches(sampler):
    """Test handling multiple sequences with mismatches"""
    spec_tokens = [[1, 2, 3], [4, 5, 6]]
    output_tokens = [1, 2, 7, 6, 4, 8, 6, 9]  # Mismatches in both sequences

    logits = create_logits_tensor(output_tokens)
    output = sampler(logits, spec_tokens)
    expected = [[1, 2, 7], [4, 8]]  # Now returns a list - only valid tokens
    assert output == expected


@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2]], [1, 2, 3], [[1, 2, 3]]),  # Perfect match with bonus
        ([[1]], [2, 3], [[2]]),  # First mismatch - only valid token
        ([[1, 2], [3, 4]], [1, 5, 6, 3, 4, 7], [[1, 5], [3, 4, 7]]),  # Mixed matches
    ],
)
def test_parametrized_cases(sampler, spec_tokens, output_tokens, expected):
    """Parametrized test for various matching scenarios"""
    logits = create_logits_tensor(output_tokens)
    output = sampler(logits, spec_tokens)
    assert output == expected


def test_logits_shape_handling(sampler):
    """Test handling of different logits tensor shapes"""
    spec_tokens = [[1, 2]]
    output_tokens = [1, 2, 3]
    vocab_size = 1000

    logits = create_logits_tensor(output_tokens, vocab_size)
    output = sampler(logits, spec_tokens)
    expected = [[1, 2, 3]]  # Now returns a list
    assert output == expected
    assert logits.shape[-1] == vocab_size
