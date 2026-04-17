"""Tests for grail.validator.batcher.WindowBatcher."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from grail.constants import (
    COMPLETIONS_PER_SUBMISSION,
    DIVERSITY_PREFIX_LEN,
    GROUP_SIZE,
    PROMPTS_PER_WINDOW,
)
from grail.validator.batcher import AcceptedCompletion, ProblemSlot, WindowBatcher
from grail.protocol.submission import SubmissionRequest, CompletionSubmission

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

WINDOW_START = 1000
RANDOMNESS = "deadbeefdeadbeef"
PROMPT_TEXT = "Q"
PROMPT_ID = "abc123"
GROUND_TRUTH = "42"

# Fake tokenizer: encode returns [ord(c) % 1000 for c in text],
# decode returns "".join(chr(t) for t in tokens if t < 128).

class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 1000 for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if t < 128)


class FakeEnv:
    name = "test"

    def __len__(self) -> int:
        return 100

    def get_problem(self, index: int) -> dict:
        return {"prompt": PROMPT_TEXT, "ground_truth": GROUND_TRUTH, "id": PROMPT_ID}

    def compute_reward(self, problem: dict, completion: str) -> float:
        return 1.0 if GROUND_TRUTH in completion else 0.0


def make_fake_model() -> MagicMock:
    return MagicMock()


def make_slot(slot_index: int = 0) -> ProblemSlot:
    problem = {"prompt": PROMPT_TEXT, "ground_truth": GROUND_TRUTH, "id": PROMPT_ID}
    return ProblemSlot(
        slot_index=slot_index,
        prompt_id=PROMPT_ID,
        problem=problem,
    )


def make_batcher(
    slots: list[ProblemSlot] | None = None,
    verify_commitment_proofs_fn=None,
    verify_signature_fn=None,
    verify_proof_version_fn=None,
) -> WindowBatcher:
    if slots is None:
        slots = [make_slot(i) for i in range(PROMPTS_PER_WINDOW)]

    # Default mocks: everything passes.
    if verify_commitment_proofs_fn is None:
        verify_commitment_proofs_fn = MagicMock(return_value=(True, 32, 32))
    if verify_signature_fn is None:
        verify_signature_fn = MagicMock(return_value=True)
    if verify_proof_version_fn is None:
        verify_proof_version_fn = MagicMock(return_value=True)

    return WindowBatcher(
        window_start=WINDOW_START,
        slots=slots,
        randomness=RANDOMNESS,
        env=FakeEnv(),
        model=make_fake_model(),
        tokenizer=FakeTokenizer(),
        verify_commitment_proofs_fn=verify_commitment_proofs_fn,
        verify_signature_fn=verify_signature_fn,
        verify_proof_version_fn=verify_proof_version_fn,
    )


# ---------------------------------------------------------------------------
# Token helpers
#
# PROMPT_TEXT = "Q"  → encode = [ord('Q') % 1000] = [81]
# prompt_length = 1
#
# For reward=1.0 we need "42" in decoded text.
# chr(52) = '4', chr(50) = '2' → tokens [52, 50] decode to "42".
# So tail tokens [52, 50, <filler>] → decoded includes "42" → reward=1.0
# ---------------------------------------------------------------------------

PROMPT_TOKENS = [ord(c) % 1000 for c in PROMPT_TEXT]   # [81]
PROMPT_LEN = len(PROMPT_TOKENS)                          # 1

# Diverse prefixes (DIVERSITY_PREFIX_LEN = 8): just 8 distinct token values.
_DIVERSE_STARTS = [
    [100, 101, 102, 103, 104, 105, 106, 107],
    [200, 201, 202, 203, 204, 205, 206, 207],
    [300, 301, 302, 303, 304, 305, 306, 307],
    [400, 401, 402, 403, 404, 405, 406, 407],
]

# Tail tokens that decode to contain "42": chr(52)='4', chr(50)='2'
_REWARD_TAIL = [52, 50]  # decodes to "42"


def make_tokens(diverse_idx: int) -> list[int]:
    """Build a full token list: [prompt] + [diverse_prefix_8] + [reward_tail]."""
    return PROMPT_TOKENS + _DIVERSE_STARTS[diverse_idx] + _REWARD_TAIL


def make_commit(tokens: list[int], prompt_length: int = PROMPT_LEN) -> dict:
    return {
        "tokens": tokens,
        "commitments": [{}] * len(tokens),
        "signature": "aabbcc",
        "proof_version": "v5",
        "rollout": {"prompt_length": prompt_length},
    }


def make_completions(
    diverse_starts: list[list[int]] | None = None,
    prompt_length: int = PROMPT_LEN,
) -> list[CompletionSubmission]:
    """Build 4 CompletionSubmission objects with distinct prefixes."""
    if diverse_starts is None:
        diverse_starts = _DIVERSE_STARTS
    result = []
    for i in range(COMPLETIONS_PER_SUBMISSION):
        tokens = PROMPT_TOKENS + diverse_starts[i] + _REWARD_TAIL
        commit = make_commit(tokens, prompt_length)
        result.append(CompletionSubmission(tokens=tokens, commit=commit))
    return result


def make_request(
    hotkey: str = "miner_A",
    slot_index: int = 0,
    prompt_id: str = PROMPT_ID,
    window_start: int = WINDOW_START,
    completions: list[CompletionSubmission] | None = None,
) -> SubmissionRequest:
    if completions is None:
        completions = make_completions()
    return SubmissionRequest(
        window_start=window_start,
        slot_index=slot_index,
        prompt_id=prompt_id,
        miner_hotkey=hotkey,
        completions=completions,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidSubmission:
    def test_valid_submission_accepted(self) -> None:
        """Happy path: all 4 completions verify → accepted, hotkey added, count=4."""
        batcher = make_batcher()
        req = make_request()
        resp = batcher.accept_submission(req)

        assert resp.accepted is True
        assert resp.reason == "ok"
        assert resp.slot_count == COMPLETIONS_PER_SUBMISSION
        assert resp.settled is False
        # Hotkey is tracked
        assert "miner_A" in batcher.slots[0].submitted_hotkeys

    def test_duplicate_hotkey_rejected(self) -> None:
        """Second submission from same hotkey to same slot → duplicate_submission."""
        batcher = make_batcher()
        req = make_request()
        first = batcher.accept_submission(req)
        assert first.accepted is True

        second = batcher.accept_submission(make_request(hotkey="miner_A"))
        assert second.accepted is False
        assert second.reason == "duplicate_submission"

    def test_zero_reward_still_accepted(self) -> None:
        """reward=0 does not reject; hotkey still added and slot count grows."""
        # FakeEnv returns 0.0 when "42" is NOT in the decoded text.
        # Build completions whose tail tokens don't include chr(52)+chr(50).
        no_reward_tail = [999, 998]  # both >=128 → decode to ""

        diverse = [
            [100, 101, 102, 103, 104, 105, 106, 107],
            [200, 201, 202, 203, 204, 205, 206, 207],
            [300, 301, 302, 303, 304, 305, 306, 307],
            [400, 401, 402, 403, 404, 405, 406, 407],
        ]
        completions = []
        for prefix in diverse:
            tokens = PROMPT_TOKENS + prefix + no_reward_tail
            commit = make_commit(tokens)
            completions.append(CompletionSubmission(tokens=tokens, commit=commit))

        batcher = make_batcher()
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)

        assert resp.accepted is True
        assert resp.slot_count == COMPLETIONS_PER_SUBMISSION
        # Rewards are all 0.0
        for ac in batcher.slots[0].accepted_completions:
            assert ac.reward == 0.0


class TestRejectionCases:
    def test_window_mismatch_rejected(self) -> None:
        batcher = make_batcher()
        req = make_request(window_start=WINDOW_START + 1)
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "window_mismatch"

    def test_invalid_slot_index_rejected(self) -> None:
        """slot_index >= PROMPTS_PER_WINDOW → invalid_slot."""
        batcher = make_batcher()
        # Bypass Pydantic by mutating after construction.
        req = make_request(slot_index=0)
        req.__dict__["slot_index"] = PROMPTS_PER_WINDOW  # out of range
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_slot"

    def test_settled_slot_rejects_new_submission(self) -> None:
        """Fill slot to GROUP_SIZE (8 miners × 4 completions each), 9th gets slot_full."""
        # GROUP_SIZE = 32 = 8 miners × 4 completions
        assert GROUP_SIZE == 32
        batcher = make_batcher()

        # 8 different miners, each with their own diverse prefixes
        for miner_idx in range(GROUP_SIZE // COMPLETIONS_PER_SUBMISSION):
            diverse = [
                [miner_idx * 100 + i, miner_idx * 100 + i + 1,
                 miner_idx * 100 + i + 2, miner_idx * 100 + i + 3,
                 miner_idx * 100 + i + 4, miner_idx * 100 + i + 5,
                 miner_idx * 100 + i + 6, miner_idx * 100 + i + 7]
                for i in range(COMPLETIONS_PER_SUBMISSION)
            ]
            completions = make_completions(diverse_starts=diverse)
            req = make_request(hotkey=f"miner_{miner_idx}", completions=completions)
            resp = batcher.accept_submission(req)
            assert resp.accepted is True, f"miner_{miner_idx} rejected: {resp.reason}"

        assert batcher.slots[0].settled is True

        # 9th miner should get slot_full
        diverse_9th = [
            [900 + i, 901 + i, 902 + i, 903 + i, 904 + i, 905 + i, 906 + i, 907 + i]
            for i in range(COMPLETIONS_PER_SUBMISSION)
        ]
        completions_9th = make_completions(diverse_starts=diverse_9th)
        req_9th = make_request(hotkey="miner_extra", completions=completions_9th)
        resp = batcher.accept_submission(req_9th)
        assert resp.accepted is False
        assert resp.reason == "slot_full"

    def test_prompt_mismatch_rejected(self) -> None:
        batcher = make_batcher()
        req = make_request(prompt_id="wrong_prompt_id")
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "prompt_mismatch"

    def test_wrong_completion_count_rejected(self) -> None:
        """Fewer completions than COMPLETIONS_PER_SUBMISSION → wrong_count."""
        batcher = make_batcher()
        req = make_request()
        # Force wrong count by direct mutation.
        req.__dict__["completions"] = req.completions[:2]
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "wrong_count"

    def test_diversity_violation_rejected(self) -> None:
        """2 completions share the same prefix → diversity_violation, hotkey NOT added."""
        batcher = make_batcher()
        # Completions 0 and 1 share the same diverse prefix.
        dup_diverse = [
            [100, 101, 102, 103, 104, 105, 106, 107],
            [100, 101, 102, 103, 104, 105, 106, 107],  # duplicate
            [300, 301, 302, 303, 304, 305, 306, 307],
            [400, 401, 402, 403, 404, 405, 406, 407],
        ]
        completions = make_completions(diverse_starts=dup_diverse)
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)

        assert resp.accepted is False
        assert resp.reason == "diversity_violation"
        # Hotkey must NOT be consumed — miner can retry.
        assert "miner_A" not in batcher.slots[0].submitted_hotkeys

    def test_cross_miner_duplicate_prefix_rejected(self) -> None:
        """Second miner reusing the first miner's prefixes → duplicate_prefix.

        This is the sybil-copy block: a second hotkey can't claim emission
        credit by submitting the same answer (or any answer with the same
        first-N generated tokens) as someone already in the slot.
        """
        batcher = make_batcher()
        # First miner: default prefixes, accepted.
        first = batcher.accept_submission(make_request(hotkey="miner_A"))
        assert first.accepted is True
        assert len(batcher.slots[0].accepted_prefixes) == COMPLETIONS_PER_SUBMISSION

        # Second miner with a different hotkey but the SAME prefixes.
        second = batcher.accept_submission(make_request(hotkey="miner_B"))
        assert second.accepted is False
        assert second.reason == "duplicate_prefix"
        # Slot count unchanged; hotkey NOT consumed (miner_B can retry).
        assert batcher.slots[0].count == COMPLETIONS_PER_SUBMISSION
        assert "miner_B" not in batcher.slots[0].submitted_hotkeys

    def test_cross_miner_partial_overlap_rejects_whole_batch(self) -> None:
        """One overlapping prefix in a batch of 4 → whole batch rejected.

        Same all-or-nothing semantics as the proof verification: if any
        completion collides with the slot's history, the miner doesn't get
        credit for the other three either.
        """
        batcher = make_batcher()
        first = batcher.accept_submission(make_request(hotkey="miner_A"))
        assert first.accepted is True

        # Build a batch where prefix #2 collides with miner_A's prefix #2,
        # but #0, #1, #3 are fresh.
        partial = [
            [500, 501, 502, 503, 504, 505, 506, 507],   # fresh
            [600, 601, 602, 603, 604, 605, 606, 607],   # fresh
            _DIVERSE_STARTS[2],                          # collides
            [700, 701, 702, 703, 704, 705, 706, 707],   # fresh
        ]
        resp = batcher.accept_submission(
            make_request(hotkey="miner_B", completions=make_completions(diverse_starts=partial))
        )
        assert resp.accepted is False
        assert resp.reason == "duplicate_prefix"
        assert batcher.slots[0].count == COMPLETIONS_PER_SUBMISSION

    def test_accepted_prefixes_tracked_per_slot(self) -> None:
        """A successful accept extends slot.accepted_prefixes by exactly 4 entries."""
        batcher = make_batcher()
        assert batcher.slots[0].accepted_prefixes == set()

        batcher.accept_submission(make_request(hotkey="miner_A"))
        assert len(batcher.slots[0].accepted_prefixes) == COMPLETIONS_PER_SUBMISSION

        # Other slots untouched.
        for s in batcher.slots[1:]:
            assert s.accepted_prefixes == set()

    def test_token_mismatch_between_top_and_commit_rejected(self) -> None:
        """c.tokens != c.commit['tokens'] → token_mismatch."""
        batcher = make_batcher()
        completions = make_completions()
        # Tamper: top-level tokens differ from commit tokens on completion 0.
        bad_tokens = completions[0].tokens[:]
        bad_tokens.append(9999)
        completions[0] = CompletionSubmission(
            tokens=bad_tokens,
            commit=completions[0].commit,  # commit still has old tokens
        )
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)

        assert resp.accepted is False
        assert resp.reason == "token_mismatch"

    def test_invalid_proof_version_rejected(self) -> None:
        """verify_proof_version_fn returns False → invalid_proof_version."""
        bad_version = MagicMock(return_value=False)
        batcher = make_batcher(verify_proof_version_fn=bad_version)
        req = make_request()
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_proof_version"

    def test_invalid_signature_rejected(self) -> None:
        """verify_signature_fn returns False → invalid_signature."""
        bad_sig = MagicMock(return_value=False)
        batcher = make_batcher(verify_signature_fn=bad_sig)
        req = make_request()
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_signature"

    def test_invalid_prompt_rejected(self) -> None:
        """Completion tokens don't start with the expected prompt tokens → invalid_prompt."""
        batcher = make_batcher()
        # Replace prompt tokens with something wrong.
        wrong_prompt = [9999]  # not [81]
        diverse = _DIVERSE_STARTS
        completions = []
        for i in range(COMPLETIONS_PER_SUBMISSION):
            tokens = wrong_prompt + diverse[i] + _REWARD_TAIL
            commit = make_commit(tokens, prompt_length=1)  # declares prompt_length=1
            completions.append(CompletionSubmission(tokens=tokens, commit=commit))
        req = make_request(completions=completions)
        resp = batcher.accept_submission(req)
        assert resp.accepted is False
        assert resp.reason == "invalid_prompt"

    def test_invalid_proof_rejects_whole_batch(self) -> None:
        """Proof fails on 3rd completion → whole batch rejected, slot count unchanged."""
        call_count = [0]

        def selective_proof_fail(commit, model, randomness):
            call_count[0] += 1
            # Fail on the 3rd call.
            if call_count[0] == 3:
                return (False, 0, 32)
            return (True, 32, 32)

        batcher = make_batcher(verify_commitment_proofs_fn=selective_proof_fail)
        req = make_request()
        resp = batcher.accept_submission(req)

        assert resp.accepted is False
        assert resp.reason == "invalid_proof"
        assert batcher.slots[0].count == 0  # no completions added
        assert "miner_A" not in batcher.slots[0].submitted_hotkeys


class TestWindowState:
    def test_is_window_complete_only_when_all_slots_settled(self) -> None:
        """Settle 7 slots → False; settle 8th → True."""
        batcher = make_batcher()
        assert batcher.is_window_complete() is False

        # Settle slots 0..6 by filling them with GROUP_SIZE completions each.
        for slot_idx in range(PROMPTS_PER_WINDOW - 1):
            for miner_idx in range(GROUP_SIZE // COMPLETIONS_PER_SUBMISSION):
                diverse = [
                    [slot_idx * 1000 + miner_idx * 10 + i,
                     slot_idx * 1000 + miner_idx * 10 + i + 1,
                     slot_idx * 1000 + miner_idx * 10 + i + 2,
                     slot_idx * 1000 + miner_idx * 10 + i + 3,
                     slot_idx * 1000 + miner_idx * 10 + i + 4,
                     slot_idx * 1000 + miner_idx * 10 + i + 5,
                     slot_idx * 1000 + miner_idx * 10 + i + 6,
                     slot_idx * 1000 + miner_idx * 10 + i + 7]
                    for i in range(COMPLETIONS_PER_SUBMISSION)
                ]
                completions = make_completions(diverse_starts=diverse)
                req = make_request(
                    hotkey=f"miner_{slot_idx}_{miner_idx}",
                    slot_index=slot_idx,
                    completions=completions,
                )
                resp = batcher.accept_submission(req)
                assert resp.accepted is True

        assert batcher.is_window_complete() is False

        # Settle last slot (7).
        last_slot = PROMPTS_PER_WINDOW - 1
        for miner_idx in range(GROUP_SIZE // COMPLETIONS_PER_SUBMISSION):
            diverse = [
                [8000 + miner_idx * 10 + i,
                 8000 + miner_idx * 10 + i + 1,
                 8000 + miner_idx * 10 + i + 2,
                 8000 + miner_idx * 10 + i + 3,
                 8000 + miner_idx * 10 + i + 4,
                 8000 + miner_idx * 10 + i + 5,
                 8000 + miner_idx * 10 + i + 6,
                 8000 + miner_idx * 10 + i + 7]
                for i in range(COMPLETIONS_PER_SUBMISSION)
            ]
            completions = make_completions(diverse_starts=diverse)
            req = make_request(
                hotkey=f"miner_last_{miner_idx}",
                slot_index=last_slot,
                completions=completions,
            )
            resp = batcher.accept_submission(req)
            assert resp.accepted is True

        assert batcher.is_window_complete() is True

    def test_get_window_state_reflects_counts(self) -> None:
        """After submitting 4 completions to slot 0, state shows count=4 for slot 0."""
        batcher = make_batcher()
        req = make_request(slot_index=0)
        batcher.accept_submission(req)

        state = batcher.get_window_state()
        assert state.window_start == WINDOW_START
        assert len(state.slot_states) == PROMPTS_PER_WINDOW

        slot0 = next(s for s in state.slot_states if s.slot_index == 0)
        assert slot0.count == COMPLETIONS_PER_SUBMISSION
        assert slot0.settled is False

        for s in state.slot_states:
            if s.slot_index != 0:
                assert s.count == 0
                assert s.settled is False

    def test_is_slot_settled(self) -> None:
        batcher = make_batcher()
        assert batcher.is_slot_settled(0) is False

        # Fill it.
        for miner_idx in range(GROUP_SIZE // COMPLETIONS_PER_SUBMISSION):
            diverse = [
                [miner_idx * 50 + i, miner_idx * 50 + i + 1,
                 miner_idx * 50 + i + 2, miner_idx * 50 + i + 3,
                 miner_idx * 50 + i + 4, miner_idx * 50 + i + 5,
                 miner_idx * 50 + i + 6, miner_idx * 50 + i + 7]
                for i in range(COMPLETIONS_PER_SUBMISSION)
            ]
            completions = make_completions(diverse_starts=diverse)
            req = make_request(hotkey=f"m_{miner_idx}", completions=completions)
            batcher.accept_submission(req)

        assert batcher.is_slot_settled(0) is True


class TestScoresAndArchive:
    def test_get_miner_scores_sums_rewards(self) -> None:
        """Scores are summed across all accepted completions."""
        batcher = make_batcher()

        # miner_A submits to slot 0 (4 completions, reward=1.0 each → total=4.0)
        batcher.accept_submission(make_request(hotkey="miner_A", slot_index=0))

        # miner_B submits to slot 0 (different diverse prefixes)
        diverse_B = [
            [500 + i, 501 + i, 502 + i, 503 + i, 504 + i, 505 + i, 506 + i, 507 + i]
            for i in range(COMPLETIONS_PER_SUBMISSION)
        ]
        completions_B = make_completions(diverse_starts=diverse_B)
        batcher.accept_submission(
            make_request(hotkey="miner_B", slot_index=0, completions=completions_B)
        )

        scores = batcher.get_miner_scores()
        assert scores["miner_A"] == pytest.approx(4.0)
        assert scores["miner_B"] == pytest.approx(4.0)

    def test_get_archive_data_shape(self) -> None:
        """Archive data has correct top-level keys; empty slots are skipped."""
        batcher = make_batcher()
        batcher.accept_submission(make_request(slot_index=0))

        archive = batcher.get_archive_data()
        assert archive["window_start"] == WINDOW_START
        assert archive["randomness"] == RANDOMNESS
        assert archive["environment"] == "test"
        assert "slots" in archive

        # Only slot 0 has completions; others are empty → skipped.
        assert len(archive["slots"]) == 1
        slot_entry = archive["slots"][0]
        assert slot_entry["slot_index"] == 0
        assert slot_entry["prompt_id"] == PROMPT_ID
        assert "prompt" in slot_entry
        assert "ground_truth" in slot_entry
        assert "settled" in slot_entry
        assert len(slot_entry["completions"]) == COMPLETIONS_PER_SUBMISSION

        comp = slot_entry["completions"][0]
        assert "miner_hotkey" in comp
        assert "tokens" in comp
        assert "completion_text" in comp
        assert "reward" in comp
