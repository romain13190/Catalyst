"""Validator main loop — continuous polling, batch verification, weight submission."""

import asyncio
import hashlib
import logging
import random
from collections import defaultdict

from grail.constants import (
    BATCH_FAILURE_THRESHOLD,
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    POLL_INTERVAL_SECONDS,
    ROLLOUT_SAMPLE_MIN,
    ROLLOUT_SAMPLE_RATE,
    VERIFICATION_BATCH_SIZE,
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
    """Continuous validator — polls S3, verifies in batches, gates on failure."""

    def __init__(self, wallet, model, tokenizer, dataset, netuid: int, use_drand: bool = True):
        self.wallet = wallet
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.netuid = netuid
        self.use_drand = use_drand

        self._last_processed_window: int = -1
        self._miner_metrics: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"valid": 0, "unique": 0, "checked": 0, "total": 0}
        )
        self._windows_in_interval: int = 0

    async def run(self, subtensor):
        """Main validation loop — poll continuously."""
        logger.info(
            "Starting validation service (netuid=%d, use_drand=%s, poll=%ds)",
            self.netuid, self.use_drand, POLL_INTERVAL_SECONDS,
        )

        while True:
            try:
                current_block = await chain.get_current_block(subtensor)
                target_window = self._compute_target_window(current_block)

                if target_window <= self._last_processed_window:
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
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
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _process_window(self, subtensor, target_window: int):
        """Process a window: discover miners, verify each as files appear."""
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

        # Sample which miners to validate this window
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

        # Process each miner independently as their file becomes available
        miner_valid_indices: dict[str, set[int]] = {}
        miner_upload_times: dict[str, float | None] = {}
        miner_totals: dict[str, int] = {}

        pending = set(selected)
        attempts = 0
        max_attempts = 30  # ~5 minutes at 10s poll

        while pending and attempts < max_attempts:
            # Try all pending miners in parallel
            tasks = {
                hotkey: storage.download_window_rollouts(hotkey, target_window)
                for hotkey in pending
            }
            results = {}
            for hotkey, coro in tasks.items():
                results[hotkey] = await coro

            found_this_round = set()
            for hotkey, (rollouts, upload_time) in results.items():
                if rollouts is None:
                    continue

                found_this_round.add(hotkey)

                valid_indices, total = await self._verify_miner(
                    hotkey, rollouts, randomness, rng,
                )

                miner_valid_indices[hotkey] = valid_indices
                miner_upload_times[hotkey] = upload_time
                miner_totals[hotkey] = total

            pending -= found_this_round

            if pending:
                logger.debug(
                    "%d miners still pending, polling in %ds",
                    len(pending), POLL_INTERVAL_SECONDS,
                )
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                attempts += 1

        if pending:
            logger.info(
                "%d miners never uploaded for window %d: %s",
                len(pending), target_window,
                [hk[:8] for hk in pending],
            )

        # Cross-miner copycat detection
        copycat_submissions = {
            hotkey: {
                "indices": miner_valid_indices.get(hotkey, set()),
                "upload_time": miner_upload_times.get(hotkey),
            }
            for hotkey in miner_valid_indices
        }
        rejected_indices = detect_index_copycats(copycat_submissions)

        # Score each miner
        for hotkey in miner_valid_indices:
            valid_indices = miner_valid_indices[hotkey]
            rejected = rejected_indices.get(hotkey, set())
            total = miner_totals.get(hotkey, 0)
            final_unique = len(valid_indices - rejected)

            if rejected:
                logger.warning(
                    "Miner %s: %d indices rejected by copycat detection",
                    hotkey[:8], len(rejected),
                )

            self._miner_metrics[hotkey]["valid"] += final_unique
            self._miner_metrics[hotkey]["unique"] += final_unique
            self._miner_metrics[hotkey]["checked"] += len(valid_indices)
            self._miner_metrics[hotkey]["total"] += total

    async def _verify_miner(
        self,
        hotkey: str,
        rollouts: list[dict],
        randomness: str,
        rng: random.Random,
    ) -> tuple[set[int], int]:
        """Verify a miner's rollouts with sampling and early gating.

        1. Deduplicate rollouts by dataset_index (first occurrence wins).
        2. Sample ROLLOUT_SAMPLE_RATE of unique rollouts to verify.
        3. Verify in batches of VERIFICATION_BATCH_SIZE.
        4. If failure rate exceeds BATCH_FAILURE_THRESHOLD in any batch → gate.

        Returns:
            (valid_indices, total_unique_submitted)
            If gated, valid_indices is empty.
        """
        # Deduplicate by dataset_index — keep first occurrence only
        seen_indices: set[int] = set()
        unique_rollouts: list[dict] = []
        for rollout in rollouts:
            dataset_index = rollout.get("dataset_index")
            if dataset_index is None:
                continue
            if dataset_index in seen_indices:
                continue
            seen_indices.add(dataset_index)
            unique_rollouts.append(rollout)

        total_unique = len(unique_rollouts)
        if total_unique == 0:
            logger.info("Miner %s: no rollouts with dataset_index", hotkey[:8])
            return set(), 0

        # Sample which rollouts to verify
        sample_size = max(ROLLOUT_SAMPLE_MIN, int(total_unique * ROLLOUT_SAMPLE_RATE))
        sample_size = min(sample_size, total_unique)

        sample_indices = rng.sample(range(total_unique), sample_size)
        to_verify = [unique_rollouts[i] for i in sample_indices]

        logger.info(
            "Miner %s: %d unique rollouts, verifying %d (%.0f%%)",
            hotkey[:8], total_unique, sample_size,
            100 * sample_size / total_unique,
        )

        # Verify in batches with early gating
        seen_nonces: set[int] = set()
        verified_valid = 0
        verified_total = 0
        gated = False

        for batch_start in range(0, len(to_verify), VERIFICATION_BATCH_SIZE):
            batch = to_verify[batch_start:batch_start + VERIFICATION_BATCH_SIZE]
            batch_failures = 0

            for rollout in batch:
                verified_total += 1
                is_valid, reason = verify_rollout(
                    rollout, hotkey, self.model, self.tokenizer,
                    randomness, seen_nonces, dataset=self.dataset,
                )
                if is_valid:
                    verified_valid += 1
                else:
                    batch_failures += 1
                    logger.debug(
                        "Miner %s rollout failed: %s", hotkey[:8], reason
                    )

            # Check failure rate for this batch
            batch_size = len(batch)
            if batch_size > 0 and batch_failures / batch_size > BATCH_FAILURE_THRESHOLD:
                logger.warning(
                    "Miner %s GATED: batch failure rate %.0f%% > %.0f%% "
                    "(%d/%d failed in batch, %d/%d overall)",
                    hotkey[:8],
                    100 * batch_failures / batch_size,
                    100 * BATCH_FAILURE_THRESHOLD,
                    batch_failures, batch_size,
                    verified_total - verified_valid, verified_total,
                )
                gated = True
                break

        if gated:
            return set(), total_unique

        # Extrapolate: if sampled rollouts pass, credit all unique indices
        logger.info(
            "Miner %s: %d/%d verified passed — crediting %d unique indices",
            hotkey[:8], verified_valid, verified_total, total_unique,
        )
        return seen_indices, total_unique

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
