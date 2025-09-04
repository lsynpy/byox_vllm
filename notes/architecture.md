# Final Architecture

```cpp
Qwen3ForCausalLM
├── Qwen3Model
│   ├── Embedding (VocabParallelEmbedding)
│   ├── Qwen3DecoderLayer [x N]
│   │   ├── Input RMSNorm
│   │   ├── Qwen3Attention
│   │   │   ├── QKVParallelLinear (q_proj, k_proj, v_proj -> qkv_proj)
│   │   │   ├── RowParallelLinear (o_proj)
│   │   │   ├── RMSNorm (q_norm, k_norm)
│   │   │   ├── RotaryEmbedding
│   │   │   └── Attention
│   │   ├── Post-Attention RMSNorm
│   │   └── Qwen3MLP
│   │       ├── MergedColumnParallelLinear (gate_proj, up_proj -> gate_up_proj)
│   │       ├── SiluAndMul
│   │       └── RowParallelLinear (down_proj)
│   └── RMSNorm
└── ParallelLMHead
```

# byoxvllm Qwen3 Architecture

```cpp
Qwen3ForCausalLM()
├── Qwen3Model()
│   ├── embed_tokens: nn.Embedding()
│   ├── decoder_layers: nn.ModuleList()[Qwen3DecoderLayer()]
│   │   └── layer: Qwen3DecoderLayer() [x config.num_hidden_layers]
│   │       ├── input_layernorm: RMSNorm()
│   │       ├── self_attn: Qwen3Attention()
│   │       │   ├── q_proj: nn.Linear()
│   │       │   ├── k_proj: nn.Linear()
│   │       │   ├── v_proj: nn.Linear()
│   │       │   ├── rotary_emb: RotaryEmbedding()
│   │       │   ├── attn: Attention()
│   │       │   └── o_proj: nn.Linear()
│   │       ├── post_layernorm: RMSNorm()
│   │       └── mlp: Qwen3MLP()
│   │           ├── gate_proj: nn.Linear()
│   │           ├── up_proj: nn.Linear()
│   │           ├── act_fn: nn.SiLU()
│   │           └── down_proj: nn.Linear()
│   └── norm: RMSNorm()
└── lm_head: nn.Linear()
```
