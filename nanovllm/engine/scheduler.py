from collections import deque
from enum import Enum, auto

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.logging import get_logger

logger = get_logger(__name__)


class DecodeType(Enum):
    EMPTY = auto()
    PREFILL = auto()
    DECODE = auto()


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # self.speculative_config = config.speculative_config

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        logger.debug("Adding sequence %s", seq)
        self.waiting.append(seq)
        logger.debug("append %s to waiting: %s", seq, self.waiting)

    def schedule(self) -> tuple[list[Sequence | None], DecodeType]:
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
            logger.debug("%s Scheduled %s done", DecodeType.PREFILL, scheduled_seqs)
            return scheduled_seqs, DecodeType.PREFILL

        # decode
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
            logger.info("%s scheduled %s done", DecodeType.DECODE, scheduled_seqs)
            return scheduled_seqs, DecodeType.DECODE
        else:
            logger.debug("No sequences scheduled")
            return [], DecodeType.EMPTY

    def postprocess(
        self, seqs: list[Sequence], sampled_token_ids: list[list[int]], draft_token_ids: list[list[int]]
    ) -> list[bool]:
        for seq, token_ids, draft_ids in zip(seqs, sampled_token_ids, draft_token_ids):
            seq.append_tokens(token_ids)
            logger.info("appended tokens %s to %s", token_ids, seq)
            seq.set_draft_tokens(draft_ids)
            seq.prepare_input_ids_and_positions(token_ids, draft_ids)
            # FIXME: may not generate exactly max_tokens tokens
            if (not seq.ignore_eos and token_ids == self.eos) or seq.num_comupted_tokens >= seq.max_tokens:
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
