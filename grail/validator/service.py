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

        # Global index dedup: {dataset_index: hotkey} — loaded from S3 at startup
        self._used_indices: dict[int, str] = {}

    async def run(self, subtensor):
        """Main validation loop — poll continuously."""
        # Load persisted state from S3
        self._used_indices = await storage.load_used_indices()
        logger.info(
            "Starting validation service (netuid=%d, poll=%ds, %d used indices loaded)",
            self.netuid, POLL_INTERVAL_SECONDS, len(self._used_indices),
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

        # Process each miner as their file becomes available
        window_results: dict[str, dict] = {}
        pending = set(selected)
        attempts = 0
        max_attempts = 30  # ~5 minutes at 10s poll

        while pending and attempts < max_attempts:
            for hotkey in list(pending):
                rollouts, upload_time = await storage.download_window_rollouts(
                    hotkey, target_window
                )
                if rollouts is None:
                    continue

                pending.discard(hotkey)

                new_indices, total = await self._verify_miner(
                    hotkey, rollouts, randomness, rng,
                )

                # Record results
                window_results[hotkey] = {
                    "unique": len(new_indices),
                    "total": total,
                    "indices": list(new_indices),
                }

                self._miner_metrics[hotkey]["valid"] += len(new_indices)
                self._miner_metrics[hotkey]["unique"] += len(new_indices)
                self._miner_metrics[hotkey]["total"] += total

            if pending:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                attempts += 1

        if pending:
            logger.info(
                "%d miners never uploaded for window %d",
                len(pending), target_window,
            )

        # Persist state after each window
        await storage.save_used_indices(self._used_indices)
        await storage.save_window_results(target_window, window_results)

    async def _verify_miner(
        self,
        hotkey: str,
        rollouts: list[dict],
        randomness: str,
        rng: random.Random,
    ) -> tuple[set[int], int]:
        """Verify a miner's rollouts with sampling and early gating.

        1. Deduplicate rollouts by dataset_index (first occurrence wins).
        2. Reject indices already used globally (by any miner, any window).
        3. Sample ROLLOUT_SAMPLE_RATE of remaining rollouts to verify.
        4. Verify in batches of VERIFICATION_BATCH_SIZE.
        5. If failure rate exceeds BATCH_FAILURE_THRESHOLD in any batch → gate.

        Returns:
            (new_valid_indices, total_unique_submitted)
            If gated, new_valid_indices is empty.
        """
        # Deduplicate by dataset_index within this submission
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

        # Filter out indices already used globally
        fresh_rollouts = []
        already_used = 0
        for rollout in unique_rollouts:
            idx = rollout["dataset_index"]
            if idx in self._used_indices:
                already_used += 1
            else:
                fresh_rollouts.append(rollout)

        if already_used > 0:
            logger.info(
                "Miner %s: %d/%d indices already used globally, %d fresh",
                hotkey[:8], already_used, total_unique, len(fresh_rollouts),
            )

        if not fresh_rollouts:
            return set(), total_unique

        # Sample which rollouts to verify
        sample_size = max(ROLLOUT_SAMPLE_MIN, int(len(fresh_rollouts) * ROLLOUT_SAMPLE_RATE))
        sample_size = min(sample_size, len(fresh_rollouts))

        sample_indices = rng.sample(range(len(fresh_rollouts)), sample_size)
        to_verify = [fresh_rollouts[i] for i in sample_indices]

        logger.info(
            "Miner %s: %d fresh rollouts, verifying %d (%.0f%%)",
            hotkey[:8], len(fresh_rollouts), sample_size,
            100 * sample_size / len(fresh_rollouts),
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

        # All batches passed — credit all fresh indices
        new_indices = set()
        for rollout in fresh_rollouts:
            idx = rollout["dataset_index"]
            new_indices.add(idx)
            self._used_indices[idx] = hotkey

        logger.info(
            "Miner %s: %d/%d verified passed — crediting %d new indices (%d total used)",
            hotkey[:8], verified_valid, verified_total,
            len(new_indices), len(self._used_indices),
        )
        return new_indices, total_unique

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
