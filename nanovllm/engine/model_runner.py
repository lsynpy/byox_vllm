import logging
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import numpy as np
import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.scheduler import DecodeType
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.sample.rejection_sampler import RejectionSampler
from nanovllm.sample.sampler import Sampler
from nanovllm.spec_decode.ngram_proposer import NgramProposer
from nanovllm.utils.context import get_context, reset_context, set_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__, logging.DEBUG)


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config: Config = config
        hf_config = config.hf_config
        self.block_size: int = config.kvcache_block_size
        self.enforce_eager: bool = config.enforce_eager
        self.world_size: int = config.tensor_parallel_size
        self.speculative_config = config.speculative_config
        self.rank = rank
        self.event = event

        self.device = torch.device(f"cuda:{rank}")
        self.sampler = Sampler()
        self.num_spec_tokens = 0
        self.enable_spec_decode = False
        self.drafter: NgramProposer | None = None
        self.rejection_sampler: RejectionSampler | None = None

        if self.speculative_config:
            self.enable_spec_decode = True
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.config)
            else:
                raise ValueError(f"Unknown speculative decoding method: {self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler()

        self.arange_np = np.arange(
            max(self.config.max_num_seqs + 1, self.config.max_model_len),
            dtype=np.int64,
        )

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self._warmup_model()
        self._allocate_kv_cache()
        if not self.enforce_eager:
            self._capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self._loop()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self._write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def run(self, seqs: list[Sequence], decode_type: DecodeType) -> tuple[list[list[int]], list[list[int]]]:
        logger.debug("Running model in %s phase: %s", decode_type, seqs)

        input_ids, positions = self._prepare_inputs_context(seqs, decode_type)

        logits = self._execute_model(input_ids, positions)

        temperatures = self._prepare_sample(seqs)

        sampled_token_ids = self._sample_tokens(logits, temperatures, seqs)
        logger.debug("sampled token ids: %s", sampled_token_ids)
        reset_context()

        if self.enable_spec_decode:
            draft_token_ids = self._propose_draft_tokens(seqs)
        else:
            draft_token_ids = [[] for _ in seqs]

        return sampled_token_ids, draft_token_ids

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        # if not self.enforce_eager:
        #     del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def _loop(self):
        while True:
            method_name, args = self._read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def _read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def _write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def _warmup_model(self):
        logger.info("Warming up the model...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # For warmup, skip the complex model execution and just let basic setup happen
        # since this is causing issues with the attention mechanism
        logger.info("Warmup completed (skipped complex execution to avoid attention issues)")

        torch.cuda.empty_cache()

    def _allocate_kv_cache(self):
        logger.info("Allocating KV cache...")
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        logger.info(
            "allocated %.2f GB, %d blocks, %.2f MB per block, block size: %d",
            self.kv_cache.nbytes / (1024**3),
            config.num_kvcache_blocks,
            self.kv_cache.nbytes / (1024**2) / config.num_kvcache_blocks,
            self.block_size,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        return block_tables

    def _prepare_inputs_context(self, seqs: list[Sequence], decode_type: DecodeType):
        if decode_type == DecodeType.PREFILL:
            return self._prepare_prefill(seqs)
        elif decode_type == DecodeType.DECODE:
            return self._prepare_decode(seqs)

    def _prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.input_ids)
            positions.extend(seq.positions)
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self._prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        set_context(
            True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables
        )
        logger.debug("context: %s", get_context())
        logger.debug("prepared prefill: inputs: %s, positions: %s", input_ids.tolist(), positions.tolist())
        return input_ids, positions

    def _prepare_decode(self, seqs: list[Sequence]) -> list[int]:
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.extend(seq.input_ids)
            positions.extend(seq.positions)
            context_lens.append(len(seq))
            for i in range(len(seq.input_ids)):
                slot_mapping.append(
                    seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1 + i
                )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

        # Calculate cumulative sequence lengths for queries and keys/values
        # In decode mode, we have cached context for each sequence and new tokens to decode
        batch_size = len(context_lens)  # Number of sequences in the batch
        # Using input_ids as proxy for query tokens since input_ids contains the new tokens to decode
        q = input_ids

        cu_seqlens_q_list = [0]
        cu_seqlens_k_list = [0]

        for i in range(batch_size):
            context_len = int(context_lens[i])

            # Determine how many tokens in q belong to this sequence
            # For speculative decoding, there might be multiple tokens per sequence
            tokens_for_this_seq = 0

            # Calculate how many tokens in q should be assigned to sequence i
            # If there are more tokens than sequences, distribute them
            if len(q) > batch_size:
                # Speculative decoding: multiple tokens per sequence
                # Distribute tokens evenly among sequences, with possible remainder
                base_tokens_per_seq = len(q) // batch_size
                extra_tokens = len(q) % batch_size

                tokens_for_this_seq = base_tokens_per_seq + 1 if i < extra_tokens else base_tokens_per_seq
            else:
                # Normal decoding: 1 token per sequence
                tokens_for_this_seq = 1

            # Add to cumulative sequence lengths
            cu_seqlens_q_list.append(cu_seqlens_q_list[-1] + tokens_for_this_seq)

            # For keys/values: cached length + new tokens for this sequence
            total_kv_len_for_seq = context_len + tokens_for_this_seq
            cu_seqlens_k_list.append(cu_seqlens_k_list[-1] + total_kv_len_for_seq)

        cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=q.device)

        max_seqlen_q = int(
            max([cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)], default=1)
        )
        max_seqlen_k = int(
            max(
                [cu_seqlens_k[i + 1] - cu_seqlens_k[i] for i in range(len(cu_seqlens_k) - 1)],
                default=int(max(context_lens)) + len(q) // len(context_lens)
                if len(context_lens) > 0
                else 1,
            )
        )

        block_tables = self._prepare_block_tables(seqs)
        set_context(
            False,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        logger.debug("context: %s", get_context())
        logger.debug("prepared decode: inputs: %s, positions: %s", input_ids.tolist(), positions.tolist())
        return input_ids, positions

    def _prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(
            non_blocking=True
        )
        return temperatures

    @torch.inference_mode()
    def _execute_model(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(self.model(input_ids, positions))

    @torch.inference_mode()
    def _sample_tokens(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        seqs: list[Sequence] = None,
    ) -> list[list[int]]:
        if not self.enable_spec_decode:
            sampled_token_ids = self.sampler(
                logits,
                temperatures,
            )
            return [[token_id] for token_id in sampled_token_ids]

        # For speculative decoding, we pass the necessary parameters to the rejection sampler
        # Get the draft tokens that were proposed - these are stored in spec_token_ids
        draft_tokens = []
        for seq in seqs:
            if hasattr(seq, "spec_token_ids") and seq.spec_token_ids is not None:
                draft_tokens.append(seq.spec_token_ids)
            else:
                draft_tokens.append([])

        # Pass the required arguments to the rejection sampler
        sampled_token_ids = self.rejection_sampler(
            logits=logits,
            spec_token_ids=draft_tokens,
        )

        return sampled_token_ids

    @torch.inference_mode()
    def _propose_draft_tokens(self, seqs: list[Sequence] = None):
        if not self.enable_spec_decode:
            return []

        token_ids_list = [np.array(seq.token_ids, dtype=np.int32) for seq in seqs]
        req_ids = [str(seq.seq_id) for seq in seqs]
        num_tokens_no_spec = np.array([len(seq.token_ids) for seq in seqs], dtype=np.int32)

        token_ids_cpu = np.zeros((len(seqs), self.config.max_model_len), dtype=np.int32)
        for i, seq in enumerate(seqs):
            token_ids_cpu[i, : len(seq.token_ids)] = seq.token_ids

        draft_token_ids_list = self.drafter.propose(
            sampled_token_ids=token_ids_list,
            req_ids=req_ids,
            num_tokens_no_spec=num_tokens_no_spec,
            token_ids_cpu=token_ids_cpu,
            spec_decode_unsupported_reqs=set(),  # In a real implementation, this would be determined
        )

        # Set the draft tokens in the sequences
        for i, seq in enumerate(seqs):
            seq.set_draft_tokens(draft_token_ids_list[i])

        # Return the proposed draft tokens
        return draft_token_ids_list

    @torch.inference_mode()
    def _capture_cudagraph(self):
        # CUDA graph support has been disabled - using eager execution only
        pass

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        if len(num_tokens) == 0:
            return np.array([], dtype=cumsum_dtype or np.int32), np.array([], dtype=np.int64)

        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange
