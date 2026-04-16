"""Miner engine — vLLM generation + HuggingFace proof construction."""

import logging
import random
import time
from typing import Any

import torch

from grail.constants import (
    BLOCK_TIME_SECONDS,
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    WINDOW_LENGTH,
)
from grail.infrastructure import chain, storage
from grail.infrastructure.drand import get_beacon
from grail.protocol.crypto import create_proof
from grail.protocol.grail_verifier import GRAILVerifier
from grail.protocol.signatures import sign_commit_binding
from grail.shared.forward import forward_single_layer
from grail.shared.hf_compat import resolve_hidden_size

logger = logging.getLogger(__name__)

# Leave time at the end of the window for upload.
_UPLOAD_BUFFER_SECONDS = 30


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        dataset,
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    ):
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.dataset = dataset
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self._dataset_size = len(dataset)
        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    async def mine_window(
        self,
        subtensor,
        window_start: int,
        use_drand: bool = True,
    ) -> list[dict]:
        """Generate as many rollouts as possible within the window and upload."""
        block_hash = await chain.get_block_hash(subtensor, window_start)
        if use_drand:
            from grail.infrastructure.drand import get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            randomness = chain.compute_window_randomness(
                block_hash, beacon["randomness"], drand_round=beacon["round"]
            )
        else:
            randomness = chain.compute_window_randomness(block_hash)

        # Deadline: end of window minus upload buffer
        window_duration = WINDOW_LENGTH * BLOCK_TIME_SECONDS
        deadline = time.monotonic() + window_duration - _UPLOAD_BUFFER_SECONDS

        logger.info(
            "Mining window %d — generating until deadline (%.0fs budget)",
            window_start, window_duration - _UPLOAD_BUFFER_SECONDS,
        )

        all_rollouts = []
        used_indices: set[int] = set()
        nonce = 0

        while time.monotonic() < deadline:
            # Pick a random index not yet used this window
            dataset_index = random.randrange(self._dataset_size)
            if dataset_index in used_indices:
                continue
            used_indices.add(dataset_index)

            try:
                row = self.dataset[dataset_index]
                prompt_text = row.get("text", "")
                if not prompt_text:
                    continue

                rollout = self._generate_and_prove(
                    prompt_text, randomness, window_start, block_hash,
                    nonce, dataset_index,
                )
                all_rollouts.append(rollout)
                nonce += 1
            except Exception as e:
                logger.error("Rollout generation failed for index %d: %s", dataset_index, e)

        if all_rollouts:
            hotkey = self.wallet.hotkey.ss58_address
            await storage.upload_window_rollouts(
                hotkey, window_start, all_rollouts
            )
            logger.info(
                "Uploaded %d rollouts for window %d (%.1fs)",
                len(all_rollouts), window_start,
                time.monotonic() - (deadline - window_duration + _UPLOAD_BUFFER_SECONDS),
            )

        return all_rollouts

    def _generate_and_prove(
        self,
        prompt: str,
        randomness: str,
        window_start: int,
        block_hash: str,
        nonce: int,
        dataset_index: int,
    ) -> dict:
        """Generate text with vLLM, construct proof with HF."""
        # Step 1: Generate with vLLM (GPU 0)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_length = input_ids.shape[1]

        with torch.no_grad():
            outputs = self.vllm_model.generate(
                input_ids.to(self.vllm_model.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            )
        all_tokens = outputs[0].tolist()

        # Step 2: HF forward pass for proof (GPU 1)
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Step 3: Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # Step 4: Compute logprobs from HF (not vLLM — bit-identical with validator)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        token_logprobs = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Step 5: Create proof and sign
        model_name = getattr(self.hf_model, "name_or_path", "unknown")

        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        # Step 6: Package rollout
        commit = {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": "v5",
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
            },
        }

        return {
            "window_start": window_start,
            "dataset_index": dataset_index,
            "nonce": nonce,
            "block_hash": block_hash,
            "hotkey": self.wallet.hotkey.ss58_address,
            "commit": commit,
        }
