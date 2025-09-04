# TODO List for byoxvllm

## Inference

- [ ] Validate correctness by using nanovllm as reference
- [ ] Implement weight packing optimization for QKV and MLP projections
- [ ] Add tensor parallelism support for distributed inference
- [ ] Implement specialized linear layers (ColumnParallelLinear, RowParallelLinear, QKVParallelLinear)
- [ ] Add FlashAttention for optimized attention computation
- [ ] Implement KV cache management with PagedAttention
- [ ] Add block manager for memory allocation
- [ ] Implement scheduler for batch management
- [ ] Add CUDA graph support for decode phase
- [ ] Implement prefix caching for faster prompt processing
- [ ] Add torch compilation for optimized sampling
- [ ] Add residual arg to RMSNorm.forward() and Qwen3DecoderLayer.forward()
- [ ] Pre-compute and cache cos/sin values in RotaryEmbedding

## Qwen3 model

- [x] Byte-level Bye-Pair Encoding (BBPE)
- [x] Grouped Query Attention (GQA)
- [x] Rotary Positional Embeddings (RoPE)
- [x] SwiGLU
- [x] RMSNorm
- [x] QK-Norm
- [ ] Dense model vs MoE model

## Coding plan

### [x] part 1. Single token LLM inference engine

- lode weight/config from pre-trained model(qwen3-0.6b)
- use BBPE tokenizer from `transformers` pkg
- qwen3 dense model support

```yaml
  - model: Qwen3ForCausalLM
    - model: Qwen3Model
      - embed_tokens: nn.Embedding
      - decoder_layers: nn.ModuleList[Qwen3DecoderLayer]
        - layer: Qwen3DecoderLayer [x config.num_hidden_layers]
          - input_layernorm: RMSNorm
          - self_attn: Qwen3Attention
            - q_proj: nn.Linear
            - k_proj: nn.Linear
            - v_proj: nn.Linear
            - rotary_emb: RotaryEmbedding
            - attn: Attention
            - o_proj: nn.Linear
          - post_layernorm: RMSNorm
          - mlp: Qwen3MLP
            - gate_proj: nn.Linear
            - up_proj: nn.Linear
            - act_fn: nn.SiLU
            - down_proj: nn.Linear
      - norm: RMSNorm
    - lm_head: nn.Linear
```
