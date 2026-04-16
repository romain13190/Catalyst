"""Tests for used_indices expiry — old entries are purged."""

from grail.constants import WINDOW_LENGTH


class TestIndexExpiry:
    def test_old_indices_purged(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._used_indices = {}
        svc._index_windows = {}

        # Simulate indices from old window
        svc._used_indices[10] = "hk_old"
        svc._index_windows[10] = 100

        # Simulate indices from recent window
        svc._used_indices[20] = "hk_new"
        svc._index_windows[20] = 4000

        # Purge: current_window=5000, cutoff=5000 - 100*30 = 2000
        # index 10 at window 100 < 2000 -> purged
        # index 20 at window 4000 >= 2000 -> kept
        svc._purge_old_indices(current_window=5000)

        assert 10 not in svc._used_indices  # old -> purged
        assert 20 in svc._used_indices  # recent -> kept

    def test_recent_indices_survive_purge(self):
        from grail.validator.service import ValidationService

        svc = ValidationService.__new__(ValidationService)
        svc._used_indices = {}
        svc._index_windows = {}

        svc._used_indices[1] = "hk"
        svc._index_windows[1] = 3500

        # cutoff = 5000 - 3000 = 2000, index at 3500 >= 2000 -> kept
        svc._purge_old_indices(current_window=5000)
        assert 1 in svc._used_indices
