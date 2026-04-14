"""GRAIL V1 CLI — mine and validate commands."""

import asyncio
import logging
import os

import typer

app = typer.Typer(name="grail", help="GRAIL V1 — Verifiable Inference Subnet")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def mine(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run GRAIL miner."""
    setup_logging(log_level)
    logger = logging.getLogger("grail.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting GRAIL miner (network=%s, netuid=%d)", network, netuid
    )

    async def _run():
        import bittensor as bt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from grail.constants import ATTN_IMPLEMENTATION, WINDOW_LENGTH
        from grail.dataset.loader import load_dataset_cached
        from grail.infrastructure.chain import get_subtensor
        from grail.miner.engine import MiningEngine

        wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        logger.info("Loading models from %s...", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        vllm_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:0").eval()

        hf_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:1").eval()

        dataset = load_dataset_cached()
        engine = MiningEngine(
            vllm_model, hf_model, tokenizer, wallet, dataset
        )

        logger.info("Miner ready. Entering main loop.")
        last_window = -1
        while True:
            try:
                current_block = await subtensor.get_current_block()
                window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                if window_start > last_window:
                    await engine.mine_window(
                        subtensor, window_start, use_drand=use_drand
                    )
                    last_window = window_start
                await asyncio.sleep(6)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Mining error: %s", e, exc_info=True)
                await asyncio.sleep(12)

    asyncio.run(_run())


@app.command()
def validate(
    use_drand: bool = typer.Option(True, help="Use drand for randomness"),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(81, help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey: str = typer.Option("default", help="Hotkey name"),
    checkpoint: str = typer.Option(..., help="Model checkpoint path"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Run GRAIL validator."""
    setup_logging(log_level)
    logger = logging.getLogger("grail.cli")

    os.environ["BT_NETWORK"] = network
    os.environ["NETUID"] = str(netuid)

    logger.info(
        "Starting GRAIL validator (network=%s, netuid=%d)", network, netuid
    )

    async def _run():
        import bittensor as bt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from grail.constants import ATTN_IMPLEMENTATION
        from grail.dataset.loader import load_dataset_cached
        from grail.infrastructure.chain import get_subtensor
        from grail.validator.service import ValidationService

        wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        subtensor = await get_subtensor()

        logger.info("Loading model from %s...", checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:0").eval()

        dataset = load_dataset_cached()
        service = ValidationService(
            wallet, model, tokenizer, dataset, netuid, use_drand=use_drand
        )
        await service.run(subtensor)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
