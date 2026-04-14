from grail.validator.weights import compute_weights


class TestComputeWeights:
    def test_single_miner(self):
        scores = {"miner_a": {"unique": 100, "valid": 100}}
        weights = compute_weights(scores)
        assert "miner_a" in weights
        assert weights["miner_a"] > 0

    def test_more_unique_gets_more_weight(self):
        scores = {
            "miner_a": {"unique": 100, "valid": 100},
            "miner_b": {"unique": 50, "valid": 50},
        }
        weights = compute_weights(scores)
        assert weights["miner_a"] > weights["miner_b"]

    def test_superlinear(self):
        """2x unique should give much more than 2x weight (exponent=4)."""
        scores_a = {"m": {"unique": 200, "valid": 200}}
        scores_b = {"m": {"unique": 100, "valid": 100}}
        wa = compute_weights(scores_a)["m"]
        wb = compute_weights(scores_b)["m"]
        # Both are single miner so both = 1.0. Test with two miners.
        scores_c = {
            "a": {"unique": 200, "valid": 200},
            "b": {"unique": 100, "valid": 100},
        }
        w = compute_weights(scores_c)
        # 200^4 / (200^4 + 100^4) = 16e8 / 17e8 ≈ 0.941
        assert w["a"] > 0.9

    def test_zero_unique_zero_weight(self):
        scores = {"m": {"unique": 0, "valid": 100}}
        weights = compute_weights(scores)
        assert weights["m"] == 0.0

    def test_zero_valid_zero_weight(self):
        scores = {"m": {"unique": 100, "valid": 0}}
        weights = compute_weights(scores)
        assert weights["m"] == 0.0

    def test_normalizes_to_one(self):
        scores = {
            "a": {"unique": 100, "valid": 100},
            "b": {"unique": 200, "valid": 200},
            "c": {"unique": 50, "valid": 50},
        }
        weights = compute_weights(scores)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_cap_applied(self):
        scores = {
            "a": {"unique": 999999, "valid": 999999},
            "b": {"unique": 999999, "valid": 999999},
        }
        weights = compute_weights(scores)
        assert abs(weights["a"] - weights["b"]) < 1e-6
        assert abs(weights["a"] - 0.5) < 1e-6

    def test_all_zero(self):
        scores = {
            "a": {"unique": 0, "valid": 0},
            "b": {"unique": 0, "valid": 0},
        }
        weights = compute_weights(scores)
        assert all(w == 0.0 for w in weights.values())

    def test_empty_input(self):
        weights = compute_weights({})
        assert weights == {}
