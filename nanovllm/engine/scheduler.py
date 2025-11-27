from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.utils.logging import logger


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        logger.debug(f"Adding sequence {seq}")
        self.waiting.append(seq)
        logger.debug(f"waiting: {self.waiting}")

    def schedule(self) -> tuple[list[Sequence], bool]:
        logger.debug("Scheduling start ...")
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            logger.debug(f"select {seq} to schedule for prefill")
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                logger.debug(f"Max batched tokens reached, cannot schedule {seq} for prefill")
                self.waiting.rotate(-1)
                logger.debug(f"waiting: {self.waiting}")
                break
            if not self.block_manager.can_allocate(seq):
                logger.debug(f"Free blocks not enough, cannot schedule {seq} for prefill")
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            logger.debug(f"waiting: {self.waiting}")
            self.running.append(seq)
            logger.debug(f"running: {self.running}")
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            logger.debug(f"[prefill] Scheduled {scheduled_seqs} done")
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            logger.debug(f"select {seq} to schedule for decode")
            logger.debug(f"running: {self.running}")
            while not self.block_manager.can_append(seq):
                if self.running:
                    self._preempt(self.running.pop())
                    logger.debug(f"running: {self.running}")
                else:
                    self._preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            logger.debug(f"running: {self.running}")
            logger.debug(f"[decode] scheduled {scheduled_seqs} done")
            return scheduled_seqs, False
        else:
            logger.debug("No sequences scheduled")
            return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                logger.debug(f"running: {self.running}")

    def _preempt(self, seq: Sequence):
        logger.debug(f"Preempting {seq}")
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        logger.debug(f"waiting: {self.waiting}")
