import torch
from nanovllm.layers.sampler import Sampler


def test_sampler_temperature_zero():
    """Test that sampler performs greedy sampling when temperature is 0"""
    sampler = Sampler()

    # Create test logits - shape (batch_size, vocab_size)
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1], [0.1, 0.5, 2.0, 1.0]], dtype=torch.float32)

    # Test with temperature 0 (should be equivalent to argmax)
    temperatures = torch.tensor([0.0, 0.0])

    result = sampler(logits, temperatures)

    # For temperature 0, we expect argmax behavior (greedy sampling)
    expected = torch.argmax(logits, dim=-1)

    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    assert torch.equal(result, torch.tensor([0, 2]))


def test_sampler_normal_temperature():
    """Test that sampler works normally with temperature > 0"""
    sampler = Sampler()

    # Create test logits - shape (batch_size, vocab_size)
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1], [0.1, 0.5, 2.0, 1.0]], dtype=torch.float32)

    # Test with normal temperature > 0
    temperatures = torch.tensor([1.0, 0.5])

    result = sampler(logits, temperatures)

    # Result should be valid token indices (within vocab size)
    assert result.shape[0] == logits.shape[0]  # batch size matches
    assert torch.all(result >= 0)  # all indices are non-negative
    assert torch.all(result < logits.shape[1])  # all indices within vocab size


def test_sampler_mixed_temperatures():
    """Test that sampler handles mixed temperatures (some 0, some > 0)"""
    sampler = Sampler()

    # Create test logits - shape (batch_size, vocab_size)
    logits = torch.tensor(
        [[2.0, 1.0, 0.5, 0.1], [0.1, 0.5, 2.0, 1.0], [1.0, 3.0, 0.5, 0.2]], dtype=torch.float32
    )

    # Test with mixed temperatures: some 0, some > 0
    temperatures = torch.tensor([0.0, 1.0, 0.0])

    result = sampler(logits, temperatures)

    # For temperature 0 positions, we expect argmax behavior (deterministic)
    # Position 0: should be 0 (argmax of [2.0, 1.0, 0.5, 0.1])
    # Position 2: should be 1 (argmax of [1.0, 3.0, 0.5, 0.2])
    # Position 1: uses sampling with temp=1.0 (stochastic, can be any token)
    assert result[0] == 0, f"Expected token 0 for temp=0, got {result[0]}"
    assert result[2] == 1, f"Expected token 1 for temp=0, got {result[2]}"

    # Verify all results are valid token indices
    assert torch.all(result >= 0) and torch.all(result < logits.shape[1])


def test_sampler_single_temperature_zero():
    """Test sampler with single temperature 0"""
    sampler = Sampler()

    logits = torch.tensor([[1.0, 5.0, 2.0]], dtype=torch.float32)
    temperatures = torch.tensor([0.0])

    result = sampler(logits, temperatures)
    expected = torch.tensor([1])  # index of max value (5.0)

    assert torch.equal(result, expected)


def test_sampler_all_high_temperature():
    """Test sampler behavior with high temperatures"""
    sampler = Sampler()

    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]], dtype=torch.float32)
    temperatures = torch.tensor([2.0])

    result = sampler(logits, temperatures)

    # Result should be valid token index
    assert result.shape[0] == 1
    assert 0 <= result[0] < logits.shape[1]
