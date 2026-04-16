"""Validator main loop — synchronous GRPO batching via HTTP endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any

from grail.constants import (
    BLOCK_TIME_SECONDS,
    POLL_INTERVAL_SECONDS,
    PROMPTS_PER_WINDOW,
    VALIDATOR_HTTP_PORT,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
)
from grail.environment.base import Environment
from grail.infrastructure import chain, storage
from grail.miner.prompts import derive_window_prompts
from grail.validator.batcher import ProblemSlot, WindowBatcher
from grail.validator.server import ValidatorServer
from grail.validator.weights import compute_weights

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH  # 12


class ValidationService:
    def __init__(
        self,
        wallet,
        model,
        tokenizer,
        env: Environment,
        netuid: int,
        *,
        use_drand: bool = True,
        http_host: str = "0.0.0.0",
        http_port: int = VALIDATOR_HTTP_PORT,
    ) -> None:
        self.wallet = wallet
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.netuid = netuid
        self.use_drand = use_drand

        self._last_processed_window: int = -1
        self._miner_scores: defaultdict[str, float] = defaultdict(float)
        self._windows_in_interval: int = 0

        self.server = ValidatorServer(host=http_host, port=http_port)

    async def run(self, subtensor) -> None:
        await self.server.start()
        logger.info(
            "Validator started: env=%s, netuid=%d, http=%s:%d",
            self.env.name, self.netuid, self.server.host, self.server.port,
        )
        try:
            while True:
                try:
                    current_block = await chain.get_current_block(subtensor)
                    target_window = self._compute_target_window(current_block)
                    if target_window <= self._last_processed_window:
                        await asyncio.sleep(POLL_INTERVAL_SECONDS)
                        continue
                    await self._run_window(subtensor, target_window)
                    self._last_processed_window = target_window
                    self._windows_in_interval += 1
                    if self._windows_in_interval >= ROLLING_WINDOWS:
                        await self._submit_weights(subtensor)
                        self._miner_scores.clear()
                        self._windows_in_interval = 0
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Validation loop iteration failed")
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
        finally:
            await self.server.stop()

    async def _run_window(self, subtensor, target_window: int) -> None:
        randomness = await self._derive_randomness(subtensor, target_window)
        problems = derive_window_prompts(self.env, randomness, PROMPTS_PER_WINDOW)
        slots = [
            ProblemSlot(slot_index=i, prompt_id=p["id"], problem=p)
            for i, p in enumerate(problems)
        ]
        batcher = WindowBatcher(
            window_start=target_window,
            slots=slots,
            randomness=randomness,
            env=self.env,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.server.set_active_batcher(batcher)

        deadline = time.monotonic() + WINDOW_LENGTH * BLOCK_TIME_SECONDS
        try:
            while time.monotonic() < deadline:
                if batcher.is_window_complete():
                    logger.info("Window %d settled early", target_window)
                    break
                await asyncio.sleep(1)

            scores = batcher.get_miner_scores()
            for hk, s in scores.items():
                self._miner_scores[hk] += s

            archive = batcher.get_archive_data()
            try:
                await storage.upload_window_dataset(target_window, archive)
            except Exception:
                logger.exception("Failed to upload window dataset")
        finally:
            self.server.set_active_batcher(None)

    async def _derive_randomness(self, subtensor, target_window: int) -> str:
        block_hash = await chain.get_block_hash(subtensor, target_window)
        if self.use_drand:
            from grail.infrastructure.drand import get_beacon, get_current_chain
            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                target_window, chain_info["genesis_time"], chain_info["period"],
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"],
            )
        return chain.compute_window_randomness(block_hash)

    async def _submit_weights(self, subtensor) -> None:
        scores = dict(self._miner_scores)
        weights = compute_weights(scores)
        non_zero = {hk: w for hk, w in weights.items() if w > 0}
        logger.info("Submitting weights for %d miners", len(non_zero))
        for hk, w in sorted(non_zero.items(), key=lambda x: -x[1])[:10]:
            logger.info("  %s: %.6f", hk[:8], w)

        meta = await chain.get_metagraph(subtensor, self.netuid)
        hotkey_to_uid = dict(zip(meta.hotkeys, meta.uids))
        uids = []
        weight_vals = []
        for hk, w in weights.items():
            if hk in hotkey_to_uid and w > 0:
                uids.append(int(hotkey_to_uid[hk]))
                weight_vals.append(w)
        if uids:
            await chain.set_weights(
                subtensor, self.wallet, self.netuid, uids, weight_vals,
            )

    @staticmethod
    def _compute_target_window(current_block: int) -> int:
        return (current_block // WINDOW_LENGTH) * WINDOW_LENGTH - WINDOW_LENGTH
