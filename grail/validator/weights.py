"""Weight computation for GRAIL — reward-sum-based, superlinear normalization."""

from grail.constants import SUPERLINEAR_EXPONENT


def compute_weights(
    miner_scores: dict[str, float],
    superlinear_exponent: float = SUPERLINEAR_EXPONENT,
) -> dict[str, float]:
    """Normalize raw reward sums to weights via x^p.

    Args:
        miner_scores: {hotkey: cumulative reward across the rolling window interval}
        superlinear_exponent: Sybil resistance exponent (default 4.0)

    Returns:
        {hotkey: normalized_weight} summing to 1.0, or all-zero if every
        score is non-positive.
    """
    raw = {hk: max(0.0, s) ** superlinear_exponent for hk, s in miner_scores.items()}
    total = sum(raw.values())
    if total == 0:
        return {hk: 0.0 for hk in miner_scores}
    return {hk: r / total for hk, r in raw.items()}
