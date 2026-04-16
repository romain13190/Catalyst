"""Tests for rollout count and token length limits."""

from grail.constants import MAX_ROLLOUTS_PER_FILE, MAX_TOKENS_PER_ROLLOUT


class TestRolloutLimitsConstants:
    def test_max_rollouts_per_file_defined(self):
        assert isinstance(MAX_ROLLOUTS_PER_FILE, int)
        assert MAX_ROLLOUTS_PER_FILE > 0

    def test_max_tokens_per_rollout_defined(self):
        assert isinstance(MAX_TOKENS_PER_ROLLOUT, int)
        assert MAX_TOKENS_PER_ROLLOUT > 0


class TestRolloutFiltering:
    def test_excess_rollouts_truncated(self):
        """If a miner submits more than MAX_ROLLOUTS_PER_FILE, excess are dropped."""
        from grail.validator.service import _filter_rollouts

        rollouts = [
            {"dataset_index": i, "commit": {"tokens": list(range(10))}}
            for i in range(MAX_ROLLOUTS_PER_FILE + 500)
        ]
        filtered = _filter_rollouts(rollouts)
        assert len(filtered) <= MAX_ROLLOUTS_PER_FILE

    def test_oversized_tokens_rejected(self):
        """Rollouts with tokens exceeding MAX_TOKENS_PER_ROLLOUT are dropped."""
        from grail.validator.service import _filter_rollouts

        rollouts = [
            {"dataset_index": 0, "commit": {"tokens": list(range(MAX_TOKENS_PER_ROLLOUT + 1))}},
            {"dataset_index": 1, "commit": {"tokens": list(range(100))}},
        ]
        filtered = _filter_rollouts(rollouts)
        assert len(filtered) == 1
        assert filtered[0]["dataset_index"] == 1

    def test_valid_rollouts_pass_through(self):
        from grail.validator.service import _filter_rollouts

        rollouts = [
            {"dataset_index": i, "commit": {"tokens": list(range(100))}}
            for i in range(10)
        ]
        assert len(_filter_rollouts(rollouts)) == 10
