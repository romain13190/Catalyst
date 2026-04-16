from grail.validator.weights import compute_weights


class TestComputeWeights:
    def test_single_miner_gets_weight_one(self):
        weights = compute_weights({"miner_a": 42.0})
        assert abs(weights["miner_a"] - 1.0) < 1e-9

    def test_higher_reward_gets_higher_weight(self):
        weights = compute_weights({"a": 200.0, "b": 100.0})
        assert weights["a"] > weights["b"]

    def test_superlinear_bigger_miner_gets_more_than_90_pct(self):
        """200 vs 100 with exponent=4: 200^4/(200^4+100^4) ≈ 0.941."""
        weights = compute_weights({"a": 200.0, "b": 100.0})
        assert weights["a"] > 0.9

    def test_zero_reward_gives_zero_weight(self):
        weights = compute_weights({"a": 0.0, "b": 10.0})
        assert weights["a"] == 0.0

    def test_negative_reward_treated_as_zero(self):
        weights = compute_weights({"a": -5.0, "b": 10.0})
        assert weights["a"] == 0.0
        assert weights["b"] > 0.0

    def test_multiple_miners_sum_to_one(self):
        scores = {"a": 100.0, "b": 200.0, "c": 50.0}
        weights = compute_weights(scores)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_empty_input_returns_empty_dict(self):
        assert compute_weights({}) == {}

    def test_all_zeros_returns_all_zero_weights(self):
        weights = compute_weights({"a": 0.0, "b": 0.0})
        assert all(w == 0.0 for w in weights.values())
        assert set(weights.keys()) == {"a", "b"}
