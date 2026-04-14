"""Dataset loader for karpathy/climbmix-400b-shuffle."""

import logging
from typing import Any

from grail.constants import DATASET_NAME, DATASET_SPLIT

logger = logging.getLogger(__name__)


def load_dataset_cached():
    """Load dataset with HuggingFace Arrow memory-mapping (cached after first load)."""
    from datasets import load_dataset

    logger.info("Loading dataset %s (split=%s)...", DATASET_NAME, DATASET_SPLIT)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    logger.info("Dataset loaded: %d examples", len(ds))
    return ds


def get_prompt_by_index(dataset: Any, index: int) -> str | None:
    """Return the text at the given dataset index, or None if invalid.

    Args:
        dataset: HuggingFace dataset (or mock with __len__ and __getitem__).
        index: Row index into the dataset.

    Returns:
        The text string, or None if index is out of range or text is empty/missing.
    """
    if index < 0 or index >= len(dataset):
        return None
    try:
        row = dataset[index]
        text = row.get("text")
        if not text:
            return None
        return text
    except Exception:
        return None
