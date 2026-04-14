"""R2/S3 object storage for window rollout files."""

import asyncio
import gzip
import json
import logging
import os
from typing import Any

from aiobotocore.session import get_session
from botocore.config import Config

logger = logging.getLogger(__name__)

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = get_session()
    return _SESSION


def get_s3_client(
    account_id: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    bucket_name: str | None = None,
):
    """Create S3 client context for R2."""
    account_id = account_id or os.getenv("R2_ACCOUNT_ID", "")
    access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID", "")
    secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY", "")
    endpoint = os.getenv("R2_ENDPOINT_URL") or f"https://{account_id}.r2.cloudflarestorage.com"
    region = os.getenv("R2_REGION", "us-east-1")

    config = Config(
        connect_timeout=3,
        read_timeout=30,
        retries={"max_attempts": 2},
    )
    return _get_session().create_client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=config,
    )


async def upload_json(key: str, data: Any, **client_kwargs) -> bool:
    """Upload JSON data to S3."""
    payload = json.dumps(data, separators=(",", ":")).encode()
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
        await client.put_object(Bucket=bucket, Key=key, Body=payload)
    return True


async def download_json(key: str, **client_kwargs) -> dict | None:
    """Download and parse JSON from S3."""
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            if key.endswith(".gz"):
                body = gzip.decompress(body)
            return json.loads(body)
    except Exception as e:
        logger.debug("download_json failed for %s: %s", key, e)
        return None


async def file_exists(key: str, **client_kwargs) -> bool:
    """Check if file exists in S3."""
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
            await client.head_object(Bucket=bucket, Key=key)
            return True
    except Exception:
        return False


async def load_used_indices(**client_kwargs) -> dict[int, str]:
    """Load the global used-indices map from S3.

    Returns:
        {dataset_index: hotkey} for every index ever credited.
        Empty dict if file doesn't exist yet.
    """
    key = "grail/state/used_indices.json.gz"
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            body = gzip.decompress(body)
            raw = json.loads(body)
            # JSON keys are strings — convert back to int
            return {int(k): v for k, v in raw.items()}
    except Exception as e:
        logger.info("No existing used_indices found (starting fresh): %s", e)
        return {}


async def save_used_indices(used: dict[int, str], **client_kwargs) -> bool:
    """Save the global used-indices map to S3."""
    key = "grail/state/used_indices.json.gz"
    # JSON keys must be strings
    payload = json.dumps(
        {str(k): v for k, v in used.items()}, separators=(",", ":")
    ).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info("Saved %d used indices (%d bytes)", len(used), len(compressed))
    return True


async def save_window_results(
    window_start: int, results: dict, **client_kwargs
) -> bool:
    """Save validation results for a window to S3."""
    key = f"grail/results/window-{window_start}.json.gz"
    payload = json.dumps(results, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info("Saved results for window %d (%d bytes)", window_start, len(compressed))
    return True


async def upload_window_rollouts(
    hotkey: str, window_start: int, rollouts: list[dict], **client_kwargs
) -> bool:
    """Upload window rollouts as gzipped JSON."""
    key = f"grail/windows/{hotkey}-window-{window_start}.json.gz"
    payload = json.dumps(rollouts, separators=(",", ":")).encode()
    compressed = gzip.compress(payload)
    async with get_s3_client(**client_kwargs) as client:
        bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
        await client.put_object(Bucket=bucket, Key=key, Body=compressed)
    logger.info(
        "Uploaded %d rollouts for window %d (%d bytes)",
        len(rollouts), window_start, len(compressed),
    )
    return True


async def download_window_rollouts(
    hotkey: str, window_start: int, **client_kwargs
) -> tuple[list[dict] | None, float | None]:
    """Download window rollouts and return S3 LastModified timestamp.

    Returns:
        Tuple of (rollouts, upload_time_unix). upload_time is from S3
        LastModified header. Both are None if file not found.
    """
    key = f"grail/windows/{hotkey}-window-{window_start}.json.gz"
    try:
        async with get_s3_client(**client_kwargs) as client:
            bucket = client_kwargs.get("bucket_name") or os.getenv("R2_BUCKET_ID", "grail")
            resp = await client.get_object(Bucket=bucket, Key=key)
            body = await resp["Body"].read()
            if key.endswith(".gz"):
                body = gzip.decompress(body)
            data = json.loads(body)

            upload_time = None
            last_modified = resp.get("LastModified")
            if last_modified is not None:
                upload_time = last_modified.timestamp()

            if isinstance(data, list):
                return data, upload_time
            return None, None
    except Exception as e:
        logger.debug("download_window_rollouts failed for %s: %s", key, e)
        return None, None
