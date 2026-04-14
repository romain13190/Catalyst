from unittest.mock import MagicMock

from grail.validator.verifier import verify_prompt


class TestVerifyPrompt:
    def _make_dataset(self, texts: list[str]) -> MagicMock:
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=len(texts))
        ds.__getitem__ = MagicMock(side_effect=lambda i: {"text": texts[i]})
        return ds

    def _make_tokenizer(self, mapping: dict[str, list[int]]) -> MagicMock:
        tok = MagicMock()
        tok.encode = MagicMock(side_effect=lambda text, add_special_tokens=False: mapping.get(text, []))
        return tok

    def test_valid_prompt(self):
        ds = self._make_dataset(["hello world"])
        tok = self._make_tokenizer({"hello world": [10, 20, 30]})
        rollout = {
            "dataset_index": 0,
            "commit": {
                "tokens": [10, 20, 30, 40, 50, 60],
                "rollout": {"prompt_length": 3},
            },
        }
        assert verify_prompt(rollout, ds, tok) is True

    def test_wrong_tokens(self):
        ds = self._make_dataset(["hello world"])
        tok = self._make_tokenizer({"hello world": [10, 20, 30]})
        rollout = {
            "dataset_index": 0,
            "commit": {
                "tokens": [99, 99, 99, 40, 50],
                "rollout": {"prompt_length": 3},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_wrong_prompt_length(self):
        ds = self._make_dataset(["hello world"])
        tok = self._make_tokenizer({"hello world": [10, 20, 30]})
        rollout = {
            "dataset_index": 0,
            "commit": {
                "tokens": [10, 20, 40, 50],
                "rollout": {"prompt_length": 2},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_invalid_index(self):
        ds = self._make_dataset(["only one"])
        tok = self._make_tokenizer({})
        rollout = {
            "dataset_index": 999,
            "commit": {
                "tokens": [1, 2, 3],
                "rollout": {"prompt_length": 1},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_missing_dataset_index(self):
        ds = self._make_dataset(["hello"])
        tok = self._make_tokenizer({})
        rollout = {
            "commit": {
                "tokens": [1, 2],
                "rollout": {"prompt_length": 1},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False

    def test_negative_index(self):
        ds = self._make_dataset(["hello"])
        tok = self._make_tokenizer({})
        rollout = {
            "dataset_index": -1,
            "commit": {
                "tokens": [1, 2],
                "rollout": {"prompt_length": 1},
            },
        }
        assert verify_prompt(rollout, ds, tok) is False
