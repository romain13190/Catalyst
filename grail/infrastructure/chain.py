"""Bittensor chain interactions for GRAIL."""

import asyncio
import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

NETUID = int(os.getenv("NETUID", "81"))
NETWORK = os.getenv("BT_NETWORK", "finney")


async def get_subtensor():
    """Create async subtensor."""
    import bittensor as bt

    subtensor = bt.async_subtensor(network=NETWORK)
    await asyncio.wait_for(subtensor.initialize(), timeout=120.0)
    return subtensor


async def get_metagraph(subtensor, netuid: int = NETUID):
    """Get subnet metagraph."""
    return await subtensor.metagraph(netuid)


async def get_block_hash(subtensor, block_number: int) -> str:
    """Get block hash for a given block number."""
    return await subtensor.get_block_hash(block_number)


async def get_current_block(subtensor) -> int:
    """Get current block number."""
    return await subtensor.get_current_block()


async def set_weights(
    subtensor,
    wallet,
    netuid: int,
    uids: list[int],
    weights: list[float],
) -> bool:
    """Submit weights on-chain."""
    try:
        result = await subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
        )
        logger.info("set_weights result: %s", result)
        return True
    except Exception as e:
        logger.error("set_weights failed: %s", e)
        return False


def compute_window_randomness(
    block_hash: str, drand_randomness: str | None = None
) -> str:
    """Combine block hash and optional drand randomness into window randomness."""
    clean_hash = block_hash.replace("0x", "")
    if drand_randomness:
        combined = hashlib.sha256(
            bytes.fromhex(clean_hash) + bytes.fromhex(drand_randomness)
        ).hexdigest()
        return combined
    return clean_hash
