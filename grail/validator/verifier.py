"""GRAIL proof verification — hard checks + soft checks."""

import logging
from typing import Any

from grail.constants import (
    CHALLENGE_K,
    GRAIL_PROOF_VERSION,
    LAYER_INDEX,
    MAX_TOKENS_PER_ROLLOUT,
)
from grail.dataset.loader import get_prompt_by_index

logger = logging.getLogger(__name__)


def verify_prompt(rollout: dict, dataset: Any, tokenizer: Any) -> bool:
    """Hard check: rollout tokens start with the correct prompt from the dataset.

    The miner declares a dataset_index. We look up that index, tokenize the
    text, and compare against the first prompt_length tokens of the rollout.
    """
    dataset_index = rollout.get("dataset_index")
    if dataset_index is None or dataset_index < 0:
        return False

    text = get_prompt_by_index(dataset, dataset_index)
    if text is None:
        return False

    commit = rollout.get("commit", {})
    tokens = commit.get("tokens", [])
    prompt_length = commit.get("rollout", {}).get("prompt_length", 0)

    expected_tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(expected_tokens) != prompt_length:
        return False
    if len(tokens) < prompt_length:
        return False
    if tokens[:prompt_length] != expected_tokens:
        return False

    return True


def verify_signature(commit: dict, hotkey: str) -> bool:
    """Hard check: verify Ed25519 signature on commit binding."""
    from grail.protocol.signatures import verify_commit_signature

    return verify_commit_signature(commit, hotkey)


def verify_proof_version(commit: dict) -> bool:
    """Hard check: proof version must match protocol."""
    return commit.get("proof_version") == GRAIL_PROOF_VERSION


def verify_nonce_unique(nonce: int, seen_nonces: set[int]) -> bool:
    """Hard check: nonce must not be reused within a window."""
    if nonce in seen_nonces:
        return False
    seen_nonces.add(nonce)
    return True


def verify_commitment_proofs(
    commit: dict,
    model: Any,
    window_randomness: str,
) -> tuple[bool, int, int]:
    """Hard check: verify GRAIL sketch commitments against model forward pass.

    Returns:
        (all_passed, passed_count, checked_count)
    """
    import torch

    from grail.protocol.crypto import indices_from_root
    from grail.protocol.grail_verifier import GRAILVerifier
    from grail.shared.forward import forward_single_layer
    from grail.shared.hf_compat import resolve_hidden_size

    tokens = commit["tokens"]
    commitments = commit["commitments"]

    # SECURITY: Miner must provide exactly one commitment per token.
    # Otherwise they can omit commitments for positions they can't forge,
    # and the verifier would silently skip those challenges.
    seq_len = len(tokens)
    if len(commitments) != seq_len:
        logger.warning(
            "Commitment count mismatch: %d commitments for %d tokens",
            len(commitments), seq_len,
        )
        return False, 0, 0

    # SECURITY: Reject sequences that would cause GPU OOM.
    if seq_len > MAX_TOKENS_PER_ROLLOUT:
        logger.warning(
            "Token sequence too long: %d tokens (max %d)",
            seq_len, MAX_TOKENS_PER_ROLLOUT,
        )
        return False, 0, 0

    # SECURITY: Always use the validator's independently-computed randomness.
    # Never trust the miner's claimed beacon — a miner who controls the
    # randomness can predict which positions are challenged and only forge those.
    randomness = window_randomness

    hidden_dim = resolve_hidden_size(model)
    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness)

    expected_challenges = min(CHALLENGE_K, seq_len)
    challenge_indices = indices_from_root(
        tokens, randomness, seq_len, expected_challenges
    )

    input_ids = torch.tensor([tokens], device=next(model.parameters()).device)
    with torch.no_grad():
        hidden_states, _ = forward_single_layer(model, input_ids, None, LAYER_INDEX)

    hidden_states = hidden_states[0]  # Remove batch dim: [seq_len, hidden_dim]

    passed = 0
    checked = 0
    for idx in challenge_indices:
        if idx >= seq_len:
            continue
        checked += 1
        miner_commit = commitments[idx]
        validator_hidden = hidden_states[idx]
        valid, _ = verifier.verify_commitment(
            validator_hidden, miner_commit, r_vec, seq_len, idx
        )
        if valid:
            passed += 1

    # SECURITY: All expected challenge positions must be checked and pass.
    # A miner cannot benefit from having fewer positions verified.
    all_passed = passed == checked and checked >= expected_challenges
    return all_passed, passed, checked


def verify_rollout(
    rollout: dict,
    hotkey: str,
    model: Any,
    tokenizer: Any,
    window_randomness: str,
    seen_nonces: set[int],
    dataset: Any = None,
) -> tuple[bool, str]:
    """Run all hard checks on a rollout."""
    # Prompt check — must verify that the miner used the correct dataset prompt.
    # Without this, a miner can use arbitrary prompts optimized for forgery.
    if dataset is None:
        logger.warning("Dataset not provided — cannot verify prompt origin")
        return False, "no_dataset"
    if not verify_prompt(rollout, dataset, tokenizer):
        return False, "invalid_prompt"

    commit = rollout.get("commit", {})

    if not verify_signature(commit, hotkey):
        return False, "invalid_signature"

    if not verify_proof_version(commit):
        return False, "invalid_proof_version"

    nonce = rollout.get("nonce", -1)
    if not verify_nonce_unique(nonce, seen_nonces):
        return False, "duplicate_nonce"

    all_passed, passed, checked = verify_commitment_proofs(
        commit, model, window_randomness
    )
    if not all_passed:
        return False, f"proof_failed ({passed}/{checked})"

    return True, "ok"
