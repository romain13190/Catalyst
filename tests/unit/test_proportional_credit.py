"""Tests for proportional credit — only credit based on verified pass rate."""

import asyncio
import random
from unittest.mock import MagicMock, patch


class TestProportionalCredit:
    def test_partial_pass_credits_proportionally(self):
        """If 80% of sampled rollouts pass, ~80% of fresh indices are credited."""
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._used_indices = {}
        svc._index_windows = {}
        svc._last_processed_window = 0
        svc._gated_history = {}
        svc.model = MagicMock()
        svc.tokenizer = MagicMock()
        svc.dataset = MagicMock()

        # Create 100 fresh rollouts
        rollouts = [
            {"dataset_index": i, "nonce": i, "commit": {"tokens": list(range(10))}}
            for i in range(100)
        ]

        rng = random.Random(42)

        # Mock verify_rollout: 80% pass
        call_count = [0]

        def mock_verify(rollout, hotkey, model, tokenizer, randomness, seen_nonces, dataset=None):
            call_count[0] += 1
            if call_count[0] % 5 == 0:  # 20% fail
                return False, "proof_failed"
            return True, "ok"

        with patch("grail.validator.service.verify_rollout", side_effect=mock_verify):
            new_indices, total = asyncio.get_event_loop().run_until_complete(
                svc._verify_miner("hk_test", rollouts, "aabb", rng)
            )

        # Should NOT credit all 100 — should be proportional to pass rate
        assert len(new_indices) < 100
        assert len(new_indices) > 0

    def test_full_pass_credits_all(self):
        """If 100% of sampled rollouts pass, all fresh indices are credited."""
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._used_indices = {}
        svc._index_windows = {}
        svc._last_processed_window = 0
        svc._gated_history = {}
        svc.model = MagicMock()
        svc.tokenizer = MagicMock()
        svc.dataset = MagicMock()

        rollouts = [
            {"dataset_index": i, "nonce": i, "commit": {"tokens": list(range(10))}}
            for i in range(50)
        ]

        rng = random.Random(42)

        def mock_verify(rollout, hotkey, model, tokenizer, randomness, seen_nonces, dataset=None):
            return True, "ok"

        with patch("grail.validator.service.verify_rollout", side_effect=mock_verify):
            new_indices, total = asyncio.get_event_loop().run_until_complete(
                svc._verify_miner("hk_test", rollouts, "aabb", rng)
            )

        assert len(new_indices) == 50
