"""Tests for miner failure lookback — recently gated miners are excluded."""

from grail.constants import FAILURE_LOOKBACK_WINDOWS, WINDOW_LENGTH


class TestFailureLookback:
    def test_recently_gated_miner_excluded(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._gated_history = {}

        svc._record_gating("hk_bad", window=100)
        assert svc._is_miner_excluded("hk_bad", current_window=100)
        assert svc._is_miner_excluded(
            "hk_bad",
            current_window=100 + FAILURE_LOOKBACK_WINDOWS * WINDOW_LENGTH - WINDOW_LENGTH,
        )

    def test_old_gating_expires(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._gated_history = {}

        svc._record_gating("hk_old", window=100)
        # After FAILURE_LOOKBACK_WINDOWS windows, the exclusion expires
        far_future = 100 + FAILURE_LOOKBACK_WINDOWS * WINDOW_LENGTH + WINDOW_LENGTH
        assert not svc._is_miner_excluded("hk_old", current_window=far_future)

    def test_clean_miner_not_excluded(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._gated_history = {}

        assert not svc._is_miner_excluded("hk_clean", current_window=200)
