"""WindowBatcher — per-window state machine for GRAIL verifiable inference.

Ingests miner submissions, verifies them synchronously (all-or-nothing per
batch), and tracks per-slot settlement and per-miner scores.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from grail.constants import (
    COMPLETIONS_PER_CLASS,
    COMPLETIONS_PER_SUBMISSION,
    DIVERSITY_PREFIX_LEN,
    GROUP_SIZE,
    PROMPTS_PER_WINDOW,
    SLOT_DEADLINE_SECONDS,
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


def _finalize_slot(slot: "ProblemSlot") -> None:
    """Pick the balanced subset of accepted completions that will receive credit.

    Mechanics (binary rewards):
      * For each reward class present, the ``accepted_by_reward`` list is
        already in arrival order (we append as we accept).
      * Kept per class = min(count of rarest present class, COMPLETIONS_PER_CLASS).
      * Mark ``kept = True`` on the first N completions of each present class.
      * If only one class is present, kept stays 0 for every completion —
        the slot is unusable for GRPO (can't compute advantages) and its
        entire emission budget burns.

    Once kept flags are set, the existing advantage-based scorer
    (``_compute_slot_scores``) naturally pays uniform |adv|=1.0 per kept
    completion (balanced → max std). Not kept = 0 points. The scoring
    formula is unchanged; curation just determines what's eligible.

    Idempotent: calling twice is a no-op after ``slot.finalized`` is set.
    """
    if slot.finalized:
        return
    slot.finalized = True

    class_indices = slot.accepted_by_reward
    if len(class_indices) < 2:
        return  # dégénéré — nothing kept

    kept_per_class = min(
        min(len(idx) for idx in class_indices.values()),
        COMPLETIONS_PER_CLASS,
    )
    if kept_per_class == 0:
        return

    for indices in class_indices.values():
        for i in indices[:kept_per_class]:
            slot.accepted_completions[i].kept = True


def _compute_slot_scores(slot: "ProblemSlot") -> dict[str, float]:
    """Per-miner advantage contribution within a single slot.

    Score per kept completion = ``|reward - mean| / std`` (population std,
    computed over the kept subset). Sums per miner_hotkey. Returns empty
    dict when fewer than 2 kept completions OR std == 0 (dégénéré — no
    GRPO signal to pay for).

    The finalize step typically produces a class-balanced kept subset, in
    which case |adv| = 1.0 uniformly — equivalent to a flat 1-per-kept
    payout. This formula still handles future non-binary reward cases
    where the kept set may have variance in per-class reward values.
    """
    kept = [c for c in slot.accepted_completions if c.kept]
    if len(kept) < 2:
        return {}

    rewards = [c.reward for c in kept]
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n  # population variance
    if var == 0.0:
        return {}
    std = var ** 0.5

    contribs: dict[str, float] = {}
    for c in kept:
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
    arrived_at: float = 0.0  # monotonic timestamp at accept — used for "first come wins"
    kept: bool = False       # set by _finalize_slot for completions in the balanced subset


@dataclass
class ProblemSlot:
    slot_index: int
    prompt_id: str
    problem: dict  # from env.get_problem(idx)
    accepted_completions: list[AcceptedCompletion] = field(default_factory=list)
    submitted_hotkeys: set[str] = field(default_factory=set)
    # Set of token tuples, one per accepted completion, covering the first
    # DIVERSITY_PREFIX_LEN generated tokens. Used for cross-miner dedup.
    accepted_prefixes: set[tuple[int, ...]] = field(default_factory=set)
    # {reward_value: [indices into accepted_completions in arrival order]}.
    # Used by _finalize_slot to mark the first COMPLETIONS_PER_CLASS of
    # each class as kept.
    accepted_by_reward: dict[float, list[int]] = field(default_factory=dict)
    # Set to True by _finalize_slot — no more submissions accepted after this.
    finalized: bool = False

    @property
    def count(self) -> int:
        return len(self.accepted_completions)

    @property
    def settled(self) -> bool:
        """Alias for ``finalized`` — kept for API stability.

        A slot is "settled" once its kept subset has been chosen. This
        happens either because both class quotas are full (happy path)
        or because the per-slot deadline fired (timeout path).
        """
        return self.finalized

    @property
    def both_quotas_full(self) -> bool:
        """True when both reward classes have reached COMPLETIONS_PER_CLASS."""
        got_one = any(
            len(self.accepted_by_reward.get(r, [])) >= COMPLETIONS_PER_CLASS
            for r in (1.0, 0.0)
        )
        both = all(
            len(self.accepted_by_reward.get(r, [])) >= COMPLETIONS_PER_CLASS
            for r in (1.0, 0.0)
        )
        # `got_one` is only here so a future non-binary env can override the
        # quota-full predicate without rewriting this property.
        del got_one
        return both

    def quota_remaining(self, reward: float) -> int:
        """How many more of this reward class can still be accepted."""
        return COMPLETIONS_PER_CLASS - len(self.accepted_by_reward.get(reward, []))


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
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        import threading

        self.window_start = window_start
        self.slots = slots
        self.randomness = randomness
        self.env = env
        self.model = model
        self.tokenizer = tokenizer

        # Injectable clock so tests can control the slot-deadline logic.
        self._time_fn = time_fn or time.monotonic
        self._start_time = self._time_fn()

        # Single lock serialises accept_submission, finalize_due_slots,
        # and get_window_state. Accept is already GPU-serialised; the lock
        # just prevents state races (e.g., two threads passing the quota
        # check then both committing and overflowing).
        self._lock = threading.Lock()

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
        hotkey is added to ``submitted_hotkeys``. Once both reward classes
        hit their COMPLETIONS_PER_CLASS quota, the slot auto-finalises and
        no further submissions are accepted.
        """
        with self._lock:
            return self._accept_submission_locked(request)

    def _accept_submission_locked(
        self, request: SubmissionRequest
    ) -> SubmissionResponse:
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

        # 9. Per-reward-class quota check — enforces the 64/64 composition
        # target. If any class in the batch would push past its quota, the
        # WHOLE batch is rejected (miner retries with an adjusted mix).
        batch_by_reward: dict[float, int] = {}
        for c in verified:
            batch_by_reward[c.reward] = batch_by_reward.get(c.reward, 0) + 1
        for reward, n in batch_by_reward.items():
            if slot.quota_remaining(reward) < n:
                return self._reject("quota_full", slot_count=slot.count)

        # 10. All passed — commit atomically.
        now = self._time_fn()
        for c in verified:
            c.arrived_at = now
            slot.accepted_completions.append(c)
            idx = len(slot.accepted_completions) - 1
            slot.accepted_by_reward.setdefault(c.reward, []).append(idx)
        slot.submitted_hotkeys.add(request.miner_hotkey)
        slot.accepted_prefixes.update(candidate_prefixes)

        # 11. Auto-finalize on happy path: both quotas are full → the
        # balanced subset is 64+64 = 128 kept, equal arrival-order or not.
        if slot.both_quotas_full:
            _finalize_slot(slot)
            logger.info(
                "Slot %d auto-finalized at %d/%d quotas full",
                slot.slot_index,
                len(slot.accepted_by_reward.get(1.0, [])),
                len(slot.accepted_by_reward.get(0.0, [])),
            )

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
        """Return whether slot ``slot_index`` has been finalised."""
        return self.slots[slot_index].settled

    def is_window_complete(self) -> bool:
        """Return True only when all PROMPTS_PER_WINDOW slots are finalised."""
        return all(slot.settled for slot in self.slots)

    def finalize_due_slots(self, now: float | None = None) -> int:
        """Finalise every slot whose deadline has passed.

        Called periodically by the validator service loop. Returns the
        number of slots that transitioned to finalized on this call.

        The deadline is ``self._start_time + SLOT_DEADLINE_SECONDS`` —
        all slots share the same origin (window start), so in practice
        every unfinalized slot finalises on the same call.
        """
        if now is None:
            now = self._time_fn()
        deadline = self._start_time + SLOT_DEADLINE_SECONDS
        if now < deadline:
            return 0
        with self._lock:
            changed = 0
            for slot in self.slots:
                if not slot.finalized:
                    _finalize_slot(slot)
                    changed += 1
                    logger.info(
                        "Slot %d finalized at timeout "
                        "(accepted=%d, kept=%d, class1=%d, class0=%d)",
                        slot.slot_index,
                        slot.count,
                        sum(1 for c in slot.accepted_completions if c.kept),
                        len(slot.accepted_by_reward.get(1.0, [])),
                        len(slot.accepted_by_reward.get(0.0, [])),
                    )
            return changed

    def get_window_state(self) -> WindowStateResponse:
        """Return a snapshot of all slot states for the HTTP /window/{n}/state endpoint.

        Each slot includes the histogram of accepted-completion rewards so
        miners can compute quota_remaining per class and target the rare
        one. The histogram reflects accepted (pre-finalize) counts; finalize
        doesn't change it, only marks which accepted items are kept.
        """
        with self._lock:
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
        """Return ``{hotkey: cumulative |z-score| across all KEPT completions}``.

        Layering:
          * Finalize (Phase 1) marks the balanced subset as ``kept=True``.
          * Advantage scoring (Phase 2) is applied to kept completions only.

        For a class-balanced kept subset the advantage math produces
        uniform |adv|=1.0 per completion, equivalent to a flat per-kept
        payout. For a dégénéré slot (one class only → kept stays empty or
        with std=0) the slot contributes zero to every miner.
        """
        scores: dict[str, float] = {}
        for slot in self.slots:
            slot_contribs = _compute_slot_scores(slot)
            for hk, contrib in slot_contribs.items():
                scores[hk] = scores.get(hk, 0.0) + contrib
        return scores

    def get_burn_count(self) -> int:
        """Return the number of un-used slot "seats" across the window.

        Budget per window = ``PROMPTS_PER_WINDOW * GROUP_SIZE`` (1024 seats
        with the default 8 × 128 config). Kept completions consume seats;
        everything else (accepted-but-not-kept, and slots that never
        reached both classes) contributes to the burn count, which the
        validator routes to UID_BURN at weight submission time.
        """
        total_budget = PROMPTS_PER_WINDOW * GROUP_SIZE
        kept = sum(
            sum(1 for c in slot.accepted_completions if c.kept)
            for slot in self.slots
        )
        return total_budget - kept

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
