"""WindowBatcher — per-window state machine for GRAIL verifiable inference.

Ingests miner submissions, verifies them synchronously (all-or-nothing per
batch), and tracks per-slot settlement and per-miner scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from grail.constants import (
    COMPLETIONS_PER_SUBMISSION,
    DIVERSITY_PREFIX_LEN,
    GROUP_SIZE,
    PROMPTS_PER_WINDOW,
)
from grail.environment.base import Environment
from grail.protocol.submission import (
    CompletionSubmission,
    SlotState,
    SubmissionRequest,
    SubmissionResponse,
    WindowStateResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _compute_slot_scores(slot: "ProblemSlot") -> dict[str, float]:
    """Per-miner advantage contribution within a single slot.

    Score per accepted completion = ``|reward - mean| / std`` (population
    std). Sums per miner_hotkey. Returns empty dict for degenerate slots
    (fewer than 2 completions or zero variance) — they carry no GRPO signal.
    """
    completions = slot.accepted_completions
    if len(completions) < 2:
        return {}

    rewards = [c.reward for c in completions]
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n  # population variance
    if var == 0.0:
        return {}
    std = var ** 0.5

    contribs: dict[str, float] = {}
    for c in completions:
        adv = abs(c.reward - mean) / std
        contribs[c.miner_hotkey] = contribs.get(c.miner_hotkey, 0.0) + adv
    return contribs


def _reward_histogram(slot: "ProblemSlot") -> dict[str, int]:
    """Build {str(reward): count} for the slot's accepted completions.

    Keys are stringified floats (JSON requires string keys). Only reward
    values that actually appear are included — a slot with 28 corrects and
    no wrongs returns ``{"1.0": 28}``, not ``{"1.0": 28, "0.0": 0}``.
    """
    histogram: dict[str, int] = {}
    for c in slot.accepted_completions:
        key = str(c.reward)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


# ---------------------------------------------------------------------------
# Diversity check (tested independently in tests/unit/test_diversity.py)
# ---------------------------------------------------------------------------


def _prefixes_distinct(
    token_lists: list[list[int]],
    prompt_length: int,
    prefix_len: int,
) -> bool:
    """Return True iff all extracted prefixes are pairwise distinct.

    The prefix is ``tokens[prompt_length : prompt_length + prefix_len]``.

    Degenerate case: if ``prefix_len == 0``, the function returns True because
    there is no prefix constraint to enforce (vacuously satisfied).

    Returns False if any completion is shorter than ``prompt_length + prefix_len``
    (when ``prefix_len > 0``).
    """
    if prefix_len == 0:
        return True

    prefixes: list[tuple[int, ...]] = []
    for tokens in token_lists:
        end = prompt_length + prefix_len
        if len(tokens) < end:
            return False
        prefixes.append(tuple(tokens[prompt_length:end]))

    return len(prefixes) == len(set(prefixes))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AcceptedCompletion:
    miner_hotkey: str
    tokens: list[int]
    commit: dict
    reward: float
    completion_text: str  # decoded text after the prompt


@dataclass
class ProblemSlot:
    slot_index: int
    prompt_id: str
    problem: dict  # from env.get_problem(idx)
    accepted_completions: list[AcceptedCompletion] = field(default_factory=list)
    submitted_hotkeys: set[str] = field(default_factory=set)
    # Set of token tuples, one per accepted completion, covering the first
    # DIVERSITY_PREFIX_LEN generated tokens. Used for cross-miner dedup —
    # any new completion whose prefix lands in this set is rejected. Keeps
    # the dataset diverse and blocks the trivial sybil-copy attack
    # (multiple hotkeys submitting bit-identical or trivially-perturbed
    # answers to claim multiple emission shares for the same work).
    accepted_prefixes: set[tuple[int, ...]] = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.accepted_completions)

    @property
    def settled(self) -> bool:
        return self.count >= GROUP_SIZE


# ---------------------------------------------------------------------------
# WindowBatcher
# ---------------------------------------------------------------------------


class WindowBatcher:
    """Per-window state machine that accepts miner submissions and tracks settlement.

    Parameters
    ----------
    window_start:
        Block number at which this window begins.
    slots:
        Exactly ``PROMPTS_PER_WINDOW`` ProblemSlot objects, one per prompt.
    randomness:
        Hex beacon randomness string for this window, used in proof verification.
    env:
        Environment providing problem text and reward computation.
    model:
        Language model passed through to ``verify_commitment_proofs_fn``.
    tokenizer:
        Tokenizer used to encode prompt text and decode completion tokens.
    verify_commitment_proofs_fn:
        Callable matching ``verify_commitment_proofs(commit, model, randomness)
        -> (bool, int, int)``.  Defaults to the real verifier.
    verify_signature_fn:
        Callable matching ``verify_signature(commit, hotkey) -> bool``.
        Defaults to the real verifier.
    verify_proof_version_fn:
        Callable matching ``verify_proof_version(commit) -> bool``.
        Defaults to the real verifier.
    """

    def __init__(
        self,
        window_start: int,
        slots: list[ProblemSlot],
        randomness: str,
        env: Environment,
        model: Any,
        tokenizer: Any,
        verify_commitment_proofs_fn: Callable[..., tuple[bool, int, int]] | None = None,
        verify_signature_fn: Callable[[dict, str], bool] | None = None,
        verify_proof_version_fn: Callable[[dict], bool] | None = None,
    ) -> None:
        self.window_start = window_start
        self.slots = slots
        self.randomness = randomness
        self.env = env
        self.model = model
        self.tokenizer = tokenizer

        # Inject real verifiers as defaults.
        if verify_commitment_proofs_fn is None:
            from grail.validator.verifier import verify_commitment_proofs as _vcp
            verify_commitment_proofs_fn = _vcp
        if verify_signature_fn is None:
            from grail.validator.verifier import verify_signature as _vs
            verify_signature_fn = _vs
        if verify_proof_version_fn is None:
            from grail.validator.verifier import verify_proof_version as _vpv
            verify_proof_version_fn = _vpv

        self._verify_commitment_proofs = verify_commitment_proofs_fn
        self._verify_signature = verify_signature_fn
        self._verify_proof_version = verify_proof_version_fn

    # ------------------------------------------------------------------
    # Core submission entry-point
    # ------------------------------------------------------------------

    def accept_submission(self, request: SubmissionRequest) -> SubmissionResponse:
        """Atomic accept-or-reject of all COMPLETIONS_PER_SUBMISSION completions.

        Checks are performed in order; the first failure short-circuits and
        returns a SubmissionResponse with ``accepted=False`` and a stable
        ``reason`` code.  On any failure the miner hotkey is NOT consumed, so
        the miner may retry with a corrected batch.

        On full success all completions are appended to the slot and the
        hotkey is added to ``submitted_hotkeys``.
        """

        # 1. Window guard.
        if request.window_start != self.window_start:
            return self._reject("window_mismatch", slot_count=0)

        # 2. Slot index bounds.
        if not (0 <= request.slot_index < PROMPTS_PER_WINDOW):
            return self._reject("invalid_slot", slot_count=0)

        slot = self.slots[request.slot_index]

        # 3. Slot-full guard.
        if slot.settled:
            return self._reject("slot_full", slot_count=slot.count)

        # 4. Duplicate hotkey guard.
        if request.miner_hotkey in slot.submitted_hotkeys:
            return self._reject("duplicate_submission", slot_count=slot.count)

        # 5. Prompt ID match.
        if request.prompt_id != slot.prompt_id:
            return self._reject("prompt_mismatch", slot_count=slot.count)

        # 6. Completion count (defence-in-depth; Pydantic enforces this too).
        if len(request.completions) != COMPLETIONS_PER_SUBMISSION:
            return self._reject("wrong_count", slot_count=slot.count)

        # 7. Diversity: extract the prefix from each completion. The prefixes
        # must be pairwise distinct within the batch AND distinct from every
        # prefix already accepted in this slot (cross-miner dedup).
        prompt_length = self._get_prompt_length(request.completions)
        token_lists = [c.tokens for c in request.completions]
        candidate_prefixes: list[tuple[int, ...]] = []
        for tokens in token_lists:
            end = prompt_length + DIVERSITY_PREFIX_LEN
            if len(tokens) < end:
                return self._reject("diversity_violation", slot_count=slot.count)
            candidate_prefixes.append(tuple(tokens[prompt_length:end]))

        if len(set(candidate_prefixes)) != len(candidate_prefixes):
            return self._reject("diversity_violation", slot_count=slot.count)

        if slot.accepted_prefixes.intersection(candidate_prefixes):
            return self._reject("duplicate_prefix", slot_count=slot.count)

        # 8. Per-completion verification.
        expected_prompt_tokens = self.tokenizer.encode(
            slot.problem["prompt"], add_special_tokens=False
        )

        verified: list[AcceptedCompletion] = []
        for c in request.completions:
            failure = self._verify_completion(
                c,
                request.miner_hotkey,
                slot,
                expected_prompt_tokens,
            )
            if failure is not None:
                return self._reject(failure, slot_count=slot.count)
            # Build the AcceptedCompletion.
            pl = c.commit.get("rollout", {}).get("prompt_length", 0)
            completion_text = self.tokenizer.decode(c.tokens[pl:])
            reward = self.env.compute_reward(slot.problem, completion_text)
            verified.append(
                AcceptedCompletion(
                    miner_hotkey=request.miner_hotkey,
                    tokens=c.tokens,
                    commit=c.commit,
                    reward=reward,
                    completion_text=completion_text,
                )
            )

        # 9. All passed — commit atomically.
        slot.accepted_completions.extend(verified)
        slot.submitted_hotkeys.add(request.miner_hotkey)
        slot.accepted_prefixes.update(candidate_prefixes)

        logger.debug(
            "Accepted submission from %s for slot %d (count=%d, settled=%s)",
            request.miner_hotkey,
            request.slot_index,
            slot.count,
            slot.settled,
        )

        return SubmissionResponse(
            accepted=True,
            reason="ok",
            settled=slot.settled,
            slot_count=slot.count,
        )

    # ------------------------------------------------------------------
    # Per-completion verification (returns failure reason str or None)
    # ------------------------------------------------------------------

    def _verify_completion(
        self,
        c: CompletionSubmission,
        hotkey: str,
        slot: ProblemSlot,
        expected_prompt_tokens: list[int],
    ) -> str | None:
        """Run all per-completion checks.  Returns a failure reason or None."""

        # a. Token consistency between top-level and commit.
        if c.commit.get("tokens") != c.tokens:
            return "token_mismatch"

        # b. Proof version.
        if not self._verify_proof_version(c.commit):
            return "invalid_proof_version"

        # c. Signature.
        if not self._verify_signature(c.commit, hotkey):
            return "invalid_signature"

        # d. Prompt prefix.
        prompt_length = c.commit.get("rollout", {}).get("prompt_length", 0)
        if (
            prompt_length != len(expected_prompt_tokens)
            or c.tokens[:prompt_length] != expected_prompt_tokens
        ):
            return "invalid_prompt"

        # e. GRAIL commitment proofs.
        passed, _, _ = self._verify_commitment_proofs(c.commit, self.model, self.randomness)
        if not passed:
            return "invalid_proof"

        return None

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def is_slot_settled(self, slot_index: int) -> bool:
        """Return whether slot ``slot_index`` has reached GROUP_SIZE completions."""
        return self.slots[slot_index].settled

    def is_window_complete(self) -> bool:
        """Return True only when all PROMPTS_PER_WINDOW slots are settled."""
        return all(slot.settled for slot in self.slots)

    def get_window_state(self) -> WindowStateResponse:
        """Return a snapshot of all slot states for the HTTP /window/{n}/state endpoint.

        Each slot includes the histogram of accepted-completion rewards so
        miners can choose the rare class (the one that will have the largest
        absolute advantage in the final z-score-based scoring).
        """
        slot_states = [
            SlotState(
                slot_index=slot.slot_index,
                prompt_id=slot.prompt_id,
                count=slot.count,
                settled=slot.settled,
                rewards=_reward_histogram(slot),
            )
            for slot in self.slots
        ]
        return WindowStateResponse(
            window_start=self.window_start,
            slot_states=slot_states,
        )

    def get_miner_scores(self) -> dict[str, float]:
        """Return ``{hotkey: cumulative |z-score| across all accepted completions}``.

        For each slot:
          * Compute the population mean and stdev of accepted completion rewards.
          * Each completion's contribution = ``|reward - mean| / std``.
          * Slots with fewer than 2 completions OR std == 0 contribute zero
            (no GRPO signal — degenerate slot).

        Sum contributions per miner across all slots.
        """
        scores: dict[str, float] = {}
        for slot in self.slots:
            slot_contribs = _compute_slot_scores(slot)
            for hk, contrib in slot_contribs.items():
                scores[hk] = scores.get(hk, 0.0) + contrib
        return scores

    def get_archive_data(self) -> dict:
        """Return the full dataset bundle for this window, suitable for S3 upload.

        Only slots with at least one accepted completion are included; empty
        slots are skipped.  The ``settled`` flag is preserved faithfully so
        downstream consumers know whether a slot fully settled.
        """
        slot_records = []
        for slot in self.slots:
            if not slot.accepted_completions:
                continue
            completions_data = [
                {
                    "miner_hotkey": ac.miner_hotkey,
                    "tokens": ac.tokens,
                    "completion_text": ac.completion_text,
                    "reward": ac.reward,
                }
                for ac in slot.accepted_completions
            ]
            slot_records.append(
                {
                    "slot_index": slot.slot_index,
                    "prompt_id": slot.prompt_id,
                    "prompt": slot.problem.get("prompt", ""),
                    "ground_truth": slot.problem.get("ground_truth", ""),
                    "settled": slot.settled,
                    "completions": completions_data,
                }
            )
        return {
            "window_start": self.window_start,
            "randomness": self.randomness,
            "environment": self.env.name,
            "slots": slot_records,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject(reason: str, slot_count: int) -> SubmissionResponse:
        return SubmissionResponse(
            accepted=False,
            reason=reason,
            settled=False,
            slot_count=slot_count,
        )

    @staticmethod
    def _get_prompt_length(completions: list[CompletionSubmission]) -> int:
        """Extract prompt_length from the first completion's commit rollout metadata."""
        if not completions:
            return 0
        return completions[0].commit.get("rollout", {}).get("prompt_length", 0)
