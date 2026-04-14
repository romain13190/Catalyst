from unittest.mock import MagicMock

from grail.dataset.loader import get_prompt_by_index


class TestGetPromptByIndex:
    def _make_dataset(self, texts: list[str]) -> MagicMock:
        """Create a mock dataset that supports indexing."""
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=len(texts))
        ds.__getitem__ = MagicMock(side_effect=lambda i: {"text": texts[i]})
        return ds

    def test_returns_text_at_index(self):
        ds = self._make_dataset(["hello world", "foo bar", "baz qux"])
        assert get_prompt_by_index(ds, 0) == "hello world"
        assert get_prompt_by_index(ds, 2) == "baz qux"

    def test_returns_none_for_out_of_range(self):
        ds = self._make_dataset(["only one"])
        assert get_prompt_by_index(ds, 5) is None
        assert get_prompt_by_index(ds, -1) is None

    def test_returns_none_for_missing_text_field(self):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=1)
        ds.__getitem__ = MagicMock(return_value={"other_field": "value"})
        assert get_prompt_by_index(ds, 0) is None

    def test_returns_none_for_empty_text(self):
        ds = MagicMock()
        ds.__len__ = MagicMock(return_value=1)
        ds.__getitem__ = MagicMock(return_value={"text": ""})
        assert get_prompt_by_index(ds, 0) is None
