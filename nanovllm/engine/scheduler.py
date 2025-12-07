from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__)


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
        self.speculative_config = config.speculative_config

    def is_finished(self):
        return not self.waiting and not self.running

    # For backward compatibility with tests
    def __iter__(self):
        """For backward compatibility - allows unpacking as (seqs, is_prefill)"""
        raise TypeError("'SchedulerOutput' object is not iterable")

    def add(self, seq: Sequence):
        logger.debug("Adding sequence %s", seq)
        self.waiting.append(seq)
        logger.debug("append %s to waiting: %s", seq, self.waiting)

    def schedule(self) -> SchedulerOutput:
        logger.info("Scheduling start ...")
        # prefill first
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            logger.debug("select %s from waiting", seq)
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                logger.debug("Max batched tokens reached, cannot schedule %s", seq)
                self.waiting.rotate(-1)
                logger.debug("rotate waiting with -1: %s", self.waiting)
                break
            if not self.block_manager.can_allocate(seq):
                logger.debug("Free blocks not enough, cannot schedule %s", seq)
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            logger.debug("popleft from waiting: %s", self.waiting)
            self.running.append(seq)
            logger.debug("append %s to running: %s", seq, self.running)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            if self.speculative_config:
                logger.debug("[spec_decode] Scheduled %s done", scheduled_seqs)
                return SchedulerOutput(scheduled_seqs, DecodeType.SD)
            logger.debug("[prefill] Scheduled %s done", scheduled_seqs)
            return SchedulerOutput(scheduled_seqs, DecodeType.PREFILL)

        # auto-regression and spec_decode continue
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            logger.debug("select %s from running", seq)
            logger.debug("popleft %s from running: %s", seq, self.running)
            while not self.block_manager.can_append(seq):
                if self.running:
                    tmp_seq = self.running.pop()
                    logger.debug("pop %s from running: %s", tmp_seq, self.running)
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
            logger.debug("extendleft %s to running: %s", rev_seqs, self.running)
            if self.speculative_config:
                logger.info("[spec_decode] scheduled %s done", scheduled_seqs)
                return SchedulerOutput(scheduled_seqs, DecodeType.SD)
            logger.info("[AR] scheduled %s done", scheduled_seqs)
            return SchedulerOutput(scheduled_seqs, DecodeType.AR)
        else:
            logger.debug("No sequences scheduled")
            return SchedulerOutput(None, DecodeType.EMPTY)

    def postprocess(
        self, seqs: list[Sequence], computed_token_ids: list[int], sampled_token_ids: list[int]
    ) -> list[bool]:
        for seq, token_id in zip(seqs, computed_token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_comupted_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                logger.debug("remove %s from running: %s", seq, self.running)

    def _preempt(self, seq: Sequence):
        logger.debug("Preempting %s", seq)
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        logger.debug("appendleft %s to waiting: %s", seq, self.waiting)
