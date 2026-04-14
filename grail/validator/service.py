"""Validator main loop — window processing, verification, weight submission."""

import asyncio
import hashlib
import logging
import random
import time
from collections import defaultdict

from grail.constants import (
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    WEIGHT_SUBMISSION_INTERVAL,
    WINDOW_LENGTH,
)
from grail.infrastructure import chain, storage
from grail.infrastructure.drand import get_beacon
from grail.validator.copycat import detect_index_copycats
from grail.validator.verifier import verify_rollout
from grail.validator.weights import compute_weights

logger = logging.getLogger(__name__)

ROLLING_WINDOWS = WEIGHT_SUBMISSION_INTERVAL // WINDOW_LENGTH  # 12


class ValidationService:
    """Main validator service."""

    def __init__(self, wallet, model, tokenizer, dataset, netuid: int, use_drand: bool = True):
        self.wallet = wallet
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.netuid = netuid
        self.use_drand = use_drand

        self._last_processed_window: int = -1
        self._miner_metrics: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"valid": 0, "unique": 0, "checked": 0}
        )
        self._windows_in_interval: int = 0

    async def run(self, subtensor):
        """Main validation loop."""
        logger.info(
            "Starting validation service (netuid=%d, use_drand=%s)",
            self.netuid, self.use_drand,
        )

        while True:
            try:
                current_block = await chain.get_current_block(subtensor)
                target_window = self._compute_target_window(current_block)

                if target_window <= self._last_processed_window:
                    await asyncio.sleep(6)
                    continue

                logger.info(
                    "Processing window %d (block=%d)", target_window, current_block
                )
                await self._process_window(subtensor, target_window)
                self._last_processed_window = target_window
                self._windows_in_interval += 1

                if self._windows_in_interval >= ROLLING_WINDOWS:
                    await self._submit_weights(subtensor)
                    self._miner_metrics.clear()
                    self._windows_in_interval = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Validation loop error: %s", e, exc_info=True)
                await asyncio.sleep(12)

    async def _process_window(self, subtensor, target_window: int):
        """Process a single window: fetch, verify, deduplicate, score."""
        block_hash = await chain.get_block_hash(subtensor, target_window)
        if self.use_drand:
            beacon = get_beacon(use_drand=True)
            randomness = chain.compute_window_randomness(
                block_hash, beacon["randomness"]
            )
        else:
            randomness = chain.compute_window_randomness(block_hash)

        meta = await chain.get_metagraph(subtensor, self.netuid)
        active_hotkeys = list(meta.hotkeys)

        # Sample miners deterministically
        sample_size = max(
            MINER_SAMPLE_MIN,
            min(
                int(len(active_hotkeys) * MINER_SAMPLE_RATE),
                MINER_SAMPLE_MAX,
                len(active_hotkeys),
            ),
        )
        seed = int(hashlib.sha256(block_hash.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        selected = rng.sample(
            active_hotkeys, min(sample_size, len(active_hotkeys))
        )

        logger.info(
            "Selected %d/%d miners for validation",
            len(selected), len(active_hotkeys),
        )

        # Phase 1: Download and verify each miner's rollouts
        miner_valid_rollouts: dict[str, list[dict]] = {}
        miner_valid_indices: dict[str, set[int]] = {}
        miner_upload_times: dict[str, float | None] = {}

        for hotkey in selected:
            rollouts, upload_time = await storage.download_window_rollouts(
                hotkey, target_window
            )
            if rollouts is None:
                logger.debug("No rollouts found for miner %s", hotkey)
                continue

            miner_upload_times[hotkey] = upload_time

            seen_nonces: set[int] = set()
            seen_indices: set[int] = set()
            valid_rollouts = []

            for rollout in rollouts:
                # Deduplicate indices within a single miner
                dataset_index = rollout.get("dataset_index")
                if dataset_index is not None and dataset_index in seen_indices:
                    continue

                is_valid, reason = verify_rollout(
                    rollout, hotkey, self.model, self.tokenizer,
                    randomness, seen_nonces, dataset=self.dataset,
                )
                if is_valid:
                    valid_rollouts.append(rollout)
                    if dataset_index is not None:
                        seen_indices.add(dataset_index)
                else:
                    logger.debug(
                        "Rollout failed for %s: %s", hotkey[:16], reason
                    )

            miner_valid_rollouts[hotkey] = valid_rollouts
            miner_valid_indices[hotkey] = seen_indices

            logger.info(
                "Miner %s: %d/%d valid, %d unique indices",
                hotkey[:8], len(valid_rollouts), len(rollouts), len(seen_indices),
            )

        # Phase 2: Cross-miner index dedup (copycat detection)
        copycat_submissions = {
            hotkey: {
                "indices": miner_valid_indices.get(hotkey, set()),
                "upload_time": miner_upload_times.get(hotkey),
            }
            for hotkey in miner_valid_indices
        }
        rejected_indices = detect_index_copycats(copycat_submissions)

        # Phase 3: Score — unique valid indices minus rejected ones
        for hotkey in miner_valid_indices:
            valid_indices = miner_valid_indices[hotkey]
            rejected = rejected_indices.get(hotkey, set())
            final_unique = len(valid_indices - rejected)
            final_valid = len(miner_valid_rollouts.get(hotkey, [])) - len(rejected)

            if rejected:
                logger.warning(
                    "Miner %s: %d indices rejected by copycat detection",
                    hotkey[:8], len(rejected),
                )

            self._miner_metrics[hotkey]["valid"] += max(0, final_valid)
            self._miner_metrics[hotkey]["unique"] += final_unique
            self._miner_metrics[hotkey]["checked"] += len(
                miner_valid_rollouts.get(hotkey, [])
            )

    async def _submit_weights(self, subtensor):
        """Compute and submit weights on-chain."""
        scores = dict(self._miner_metrics)
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
                subtensor, self.wallet, self.netuid, uids, weight_vals
            )

    def _compute_target_window(self, current_block: int) -> int:
        return (current_block // WINDOW_LENGTH) * WINDOW_LENGTH - WINDOW_LENGTH
