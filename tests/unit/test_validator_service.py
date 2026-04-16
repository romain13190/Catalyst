"""Unit tests for ValidationService orchestration logic."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grail.constants import WINDOW_LENGTH
from grail.validator.service import ValidationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(**kwargs) -> ValidationService:
    wallet = MagicMock()
    wallet.hotkey.ss58_address = "5Validator..."
    model = MagicMock()
    tokenizer = MagicMock()
    env = MagicMock()
    env.name = "test_env"
    env.__len__ = MagicMock(return_value=100)
    env.get_problem = MagicMock(
        side_effect=lambda idx: {
            "id": f"prob_{idx:04x}",
            "prompt": f"Question {idx}",
            "ground_truth": "42",
        }
    )
    env.compute_reward = MagicMock(return_value=1.0)

    defaults = dict(
        wallet=wallet,
        model=model,
        tokenizer=tokenizer,
        env=env,
        netuid=1,
        use_drand=False,
        http_host="127.0.0.1",
        http_port=19999,
    )
    defaults.update(kwargs)
    return ValidationService(**defaults)


# ---------------------------------------------------------------------------
# 1. _compute_target_window
# ---------------------------------------------------------------------------


class TestComputeTargetWindow:
    def test_block_95(self):
        # 95 // 30 = 3, 3*30 = 90, 90 - 30 = 60
        assert ValidationService._compute_target_window(95) == 60

    def test_block_119(self):
        # 119 // 30 = 3, 3*30 = 90, 90 - 30 = 60
        assert ValidationService._compute_target_window(119) == 60

    def test_block_120(self):
        # 120 // 30 = 4, 4*30 = 120, 120 - 30 = 90
        assert ValidationService._compute_target_window(120) == 90


# ---------------------------------------------------------------------------
# 2. _run_window settles early when batcher is complete
# ---------------------------------------------------------------------------


class TestRunWindowSettlesEarly:
    @pytest.mark.asyncio
    async def test_settles_early_accumulates_scores_and_clears_batcher(self):
        svc = _make_service()

        # Patch chain.get_block_hash to return a fixed 64-char hex string
        randomness_hex = "ab" * 32  # 64 hex chars

        fake_batcher = MagicMock()
        fake_batcher.is_window_complete.return_value = True
        fake_batcher.get_miner_scores.return_value = {"miner_a": 5.0, "miner_b": 3.0}
        fake_batcher.get_archive_data.return_value = {"window_start": 60, "slots": []}

        subtensor = MagicMock()

        with (
            patch("grail.validator.service.chain.get_block_hash", new=AsyncMock(return_value=randomness_hex)),
            patch("grail.validator.service.chain.compute_window_randomness", return_value=randomness_hex),
            patch("grail.validator.service.derive_window_prompts", return_value=[
                {"id": f"p{i}", "prompt": f"Q{i}", "ground_truth": "1"}
                for i in range(8)
            ]),
            patch("grail.validator.service.WindowBatcher", return_value=fake_batcher),
            patch("grail.validator.service.storage.upload_window_dataset", new=AsyncMock(return_value=True)),
        ):
            await svc._run_window(subtensor, 60)

        # Scores accumulated
        assert svc._miner_scores["miner_a"] == 5.0
        assert svc._miner_scores["miner_b"] == 3.0
        # Active batcher cleared
        assert svc.server.active_batcher is None


# ---------------------------------------------------------------------------
# 3. _run_window clears active batcher even on archive failure
# ---------------------------------------------------------------------------


class TestRunWindowClearsBatcherOnArchiveFailure:
    @pytest.mark.asyncio
    async def test_active_batcher_cleared_even_on_archive_exception(self):
        svc = _make_service()

        randomness_hex = "cd" * 32

        fake_batcher = MagicMock()
        fake_batcher.is_window_complete.return_value = True
        fake_batcher.get_miner_scores.return_value = {"miner_x": 2.0}
        fake_batcher.get_archive_data.return_value = {"window_start": 90, "slots": []}

        subtensor = MagicMock()

        with (
            patch("grail.validator.service.chain.get_block_hash", new=AsyncMock(return_value=randomness_hex)),
            patch("grail.validator.service.chain.compute_window_randomness", return_value=randomness_hex),
            patch("grail.validator.service.derive_window_prompts", return_value=[
                {"id": f"p{i}", "prompt": f"Q{i}", "ground_truth": "1"}
                for i in range(8)
            ]),
            patch("grail.validator.service.WindowBatcher", return_value=fake_batcher),
            patch(
                "grail.validator.service.storage.upload_window_dataset",
                new=AsyncMock(side_effect=RuntimeError("S3 unavailable")),
            ),
        ):
            # Should NOT raise — exception is caught internally
            await svc._run_window(subtensor, 90)

        # Batcher must be cleared regardless of archive failure
        assert svc.server.active_batcher is None


# ---------------------------------------------------------------------------
# 4. _submit_weights sends only non-zero miners
# ---------------------------------------------------------------------------


class TestSubmitWeightsNonZeroOnly:
    @pytest.mark.asyncio
    async def test_only_non_zero_miners_in_uids(self):
        svc = _make_service()
        svc._miner_scores = defaultdict(float, {"a": 5.0, "b": 0.0, "c": 3.0})

        meta = MagicMock()
        meta.hotkeys = ["a", "b", "c", "d"]
        meta.uids = [1, 2, 3, 4]

        set_weights_mock = AsyncMock()
        subtensor = MagicMock()

        with (
            patch("grail.validator.service.chain.get_metagraph", new=AsyncMock(return_value=meta)),
            patch("grail.validator.service.chain.set_weights", new=set_weights_mock),
        ):
            await svc._submit_weights(subtensor)

        set_weights_mock.assert_called_once()
        call_args = set_weights_mock.call_args
        submitted_uids = call_args.args[3]  # positional: subtensor, wallet, netuid, uids, weights
        assert 1 in submitted_uids  # a → uid 1
        assert 3 in submitted_uids  # c → uid 3
        assert 2 not in submitted_uids  # b → uid 2, excluded (weight 0)
        assert 4 not in submitted_uids  # d not in scores


# ---------------------------------------------------------------------------
# 5. _submit_weights skips set_weights when all scores are zero
# ---------------------------------------------------------------------------


class TestSubmitWeightsSkipsWhenAllZero:
    @pytest.mark.asyncio
    async def test_set_weights_not_called_when_all_zero(self):
        svc = _make_service()
        svc._miner_scores = defaultdict(float, {"a": 0.0})

        meta = MagicMock()
        meta.hotkeys = ["a"]
        meta.uids = [1]

        set_weights_mock = AsyncMock()
        subtensor = MagicMock()

        with (
            patch("grail.validator.service.chain.get_metagraph", new=AsyncMock(return_value=meta)),
            patch("grail.validator.service.chain.set_weights", new=set_weights_mock),
        ):
            await svc._submit_weights(subtensor)

        set_weights_mock.assert_not_called()
