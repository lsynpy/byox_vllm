from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.logging import logger


class DecodeType(Enum):
    EMPTY = auto()
    PREFILL = auto()
    AR = auto()  # Auto-regression
    SD = auto()  # Speculative decoding


@dataclass
class SchedulerOutput:
    scheduled_seqs: list[Sequence] | None
    decode_type: DecodeType
    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int] = None
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int = 0
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]] | None = None

    def __post_init__(self):
        if self.num_scheduled_tokens is None:
            self.num_scheduled_tokens = {}


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        speculative_config = config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            # Currently only supporting ngram method, so no eagle support
            # This is a simplified check - in a full implementation, there would be different
            # methods to distinguish between eagle and other speculative methods
            # For now, we'll assume eagle is not used

    def is_finished(self):
        return not self.waiting and not self.running

    # For backward compatibility with tests
    def __iter__(self):
        """For backward compatibility - allows unpacking as (seqs, is_prefill)"""
        raise TypeError("'SchedulerOutput' object is not iterable")

    def add(self, seq: Sequence):
        logger.debug(f"Adding sequence {seq}")
        self.waiting.append(seq)
        logger.debug(f"append {seq} to waiting: {self.waiting}")

    def schedule(self) -> SchedulerOutput:
        logger.debug("Scheduling start ...")
        # prefill and spec_decode
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            logger.debug(f"select {seq} from waiting")
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                logger.debug(f"Max batched tokens reached, cannot schedule {seq}")
                self.waiting.rotate(-1)
                logger.debug(f"rotate waiting with -1: {self.waiting}")
                break
            if not self.block_manager.can_allocate(seq):
                logger.debug(f"Free blocks not enough, cannot schedule {seq}")
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            logger.debug(f"popleft from waiting: {self.waiting}")
            self.running.append(seq)
            logger.debug(f"append {seq} to running: {self.running}")
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            if self.num_spec_tokens:
                logger.debug(f"[spec_decode] Scheduled {scheduled_seqs} done")
                return SchedulerOutput(scheduled_seqs, DecodeType.SD)
            logger.debug(f"[prefill] Scheduled {scheduled_seqs} done")
            return SchedulerOutput(scheduled_seqs, DecodeType.PREFILL)

        # auto-regression and spec_decode continue
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            logger.debug(f"select {seq} from running")
            logger.debug(f"popleft {seq} from running: {self.running}")
            while not self.block_manager.can_append(seq):
                if self.running:
                    tmp_seq = self.running.pop()
                    logger.debug(f"pop {tmp_seq} from running: {self.running}")
                    self._preempt(tmp_seq)
                else:
                    self._preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if scheduled_seqs:
            rev_seqs = list(reversed(scheduled_seqs))
            self.running.extendleft(rev_seqs)
            logger.debug(f"extendleft {rev_seqs} to running: {self.running}")
            if self.num_spec_tokens:
                logger.debug(f"[spec_decode] scheduled {scheduled_seqs} done")
                return SchedulerOutput(scheduled_seqs, DecodeType.SD)
            logger.debug(f"[AR] scheduled {scheduled_seqs} done")
            return SchedulerOutput(scheduled_seqs, DecodeType.AR)
        else:
            logger.debug("No sequences scheduled")
            return SchedulerOutput(None, DecodeType.EMPTY)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                logger.debug(f"remove {seq} from running: {self.running}")

    def _preempt(self, seq: Sequence):
        logger.debug(f"Preempting {seq}")
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        logger.debug(f"appendleft {seq} to waiting: {self.waiting}")
