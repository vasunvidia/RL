# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Untimed MLPerf warmup, run between setup() and the run_start stamp.

Moves one-time JIT/autotune/comm-init costs out of the timed region using
synthetic inputs only. Weights, optimizer state, LR schedule, the replay
buffer, and the real dataset are untouched.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_SEED = 20260812


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def _log(msg: str) -> None:
    print(f"[mlperf_warmup] {msg}", flush=True)


def _dry_refit(policy: Any, policy_generation: Any) -> None:
    from nemo_rl.algorithms.grpo import refit_policy_generation

    start = time.perf_counter()
    refit_policy_generation(policy, policy_generation, colocated_inference=False)
    _log(f"dry refit (unchanged init weights): {time.perf_counter() - start:.1f}s")


def _generation_prompt_lengths(
    *,
    max_model_len: int,
    chunk_tokens: int,
    max_tokens: int,
    per_engine: int,
) -> tuple[list[int], int, int]:
    """Exact prompt-token lengths at useful chunked-prefill boundaries.

    Args:
        max_model_len: Maximum combined prompt and generation length supported
            by the model.
        chunk_tokens: Maximum number of tokens in a chunked-prefill batch.
        max_tokens: Requested maximum number of tokens to generate per prompt.
        per_engine: Number of warmup requests to issue to each inference engine.

    Returns (prompt lengths, max tokens to generate, maximum legal prompt).
    """
    if max_model_len < 2:
        raise ValueError(f"max_model_len must be at least 2, got {max_model_len}")
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if per_engine < 1:
        raise ValueError(f"per_engine must be positive, got {per_engine}")

    max_tokens = max(1, min(max_tokens, max_model_len - 1))
    max_prompt = max_model_len - max_tokens
    if max_prompt < 1:
        raise ValueError(
            f"no room for a warmup prompt: max_model_len={max_model_len} "
            f"max_tokens={max_tokens}"
        )

    # A two-budget prompt cannot fit in one scheduler iteration, so it exercises
    # chunked prefill across multiple iterations. Cap it to leave room for the
    # requested generated tokens when the model context is smaller.
    long_tokens = min(2 * chunk_tokens, max_prompt)

    # Geometrically spaced sub-chunk prompts cover the range of shorter rollout
    # inputs without paying for another full chunk per request. With the
    # production 16K-token budget these are 512, 1K, 2K, and 4K tokens.
    short_cycle = [
        max(1, chunk_tokens // 32),
        max(1, chunk_tokens // 16),
        max(1, chunk_tokens // 8),
        max(1, chunk_tokens // 4),
    ]
    # Repeat those representative short shapes if an engine needs more than
    # five requests; each engine still receives the same deterministic mix.
    lengths = [long_tokens] + [
        min(short_cycle[(i - 1) % len(short_cycle)], max_prompt)
        for i in range(1, per_engine)
    ]
    return lengths, max_tokens, max_prompt


def _synthetic_prompt_token_ids(
    tokenizer: Any,
    length: int,
    *,
    engine_idx: int,
    request_idx: int,
) -> list[int]:
    """Build an exact-length, request-unique synthetic token sequence."""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    vocab_size = int(vocab_size)
    if vocab_size < 2:
        raise ValueError(f"tokenizer vocabulary is unexpectedly small: {vocab_size}")

    # Distinct primes mix the engine and request indices into different starting
    # tokens, while another prime strides through the vocabulary to avoid
    # constant-token prompts. The values have no model-specific significance.
    offset = (_SEED + engine_idx * 65537 + request_idx * 257) % vocab_size
    stride = 104729 % vocab_size or 1
    return [(offset + position * stride) % vocab_size for position in range(length)]


def _post_chat_completion(url: str, payload: dict[str, Any], timeout: float) -> int:
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        json.loads(resp.read())
        return resp.status


def _generation_warmup(
    policy_generation: Any,
    tokenizer: Any,
    policy_cfg: dict[str, Any],
    warmup_cfg: dict[str, Any],
) -> None:
    urls = [u for u in getattr(policy_generation, "dp_openai_server_base_urls", None) or [] if u]
    if not urls:
        _log("no OpenAI server URLs exposed; skipping generation warmup")
        return
    gen_cfg = policy_cfg["generation"]
    max_model_len = int(gen_cfg["vllm_cfg"]["max_model_len"])
    chunk_tokens = int(gen_cfg["vllm_kwargs"]["max_num_batched_tokens"])
    per_engine = int(
        warmup_cfg.get(
            "generation_requests_per_engine",
            os.environ.get("MLPERF_WARMUP_GEN_REQUESTS_PER_ENGINE", "16"),
        )
    )
    timeout = float(
        warmup_cfg.get(
            "generation_http_timeout_s",
            os.environ.get("MLPERF_WARMUP_GEN_TIMEOUT", "900"),
        )
    )
    max_tokens = min(
        int(
            warmup_cfg.get(
                "generation_max_tokens",
                os.environ.get("MLPERF_WARMUP_GEN_TOKENS", "192"),
            )
        ),
        int(gen_cfg["max_new_tokens"]),
    )
    prompt_lengths, max_tokens, max_prompt = _generation_prompt_lengths(
        max_model_len=max_model_len,
        chunk_tokens=chunk_tokens,
        max_tokens=max_tokens,
        per_engine=per_engine,
    )
    _log(
        "generation warmup shapes: "
        f"max_model_len={max_model_len} chunk={chunk_tokens} "
        f"max_tokens={max_tokens} max_prompt={max_prompt} "
        f"prompt_lengths={prompt_lengths}"
    )
    model_id = policy_cfg["model_name"]
    jobs = []
    for engine_idx, url in enumerate(urls):
        for req_idx, prompt_length in enumerate(prompt_lengths):
            prompt_token_ids = _synthetic_prompt_token_ids(
                tokenizer,
                prompt_length,
                engine_idx=engine_idx,
                request_idx=req_idx,
            )
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": f"MLPerf warmup {engine_idx}-{req_idx}",
                    }
                ],
                # NeMo RL's chat request extension replaces the rendered chat
                # prefix with these exact token IDs before submitting to vLLM.
                # This preserves the intended prefill shapes without relying
                # on the unsupported text-completions endpoint.
                "required_prefix_token_ids": prompt_token_ids,
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "stream": False,
            }
            jobs.append((url, payload))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(len(jobs), 512)) as pool:
        futures = [pool.submit(_post_chat_completion, u, p, timeout) for u, p in jobs]
        for future in as_completed(futures):
            future.result()
    _log(
        f"generation warmup: {len(jobs)}/{len(jobs)} requests over "
        f"{len(urls)} engines in {time.perf_counter() - start:.1f}s"
    )
    # Drop synthetic entries from the prefix cache before the timed region.
    policy_generation.finish_generation()


def _synthetic_train_batch(tokenizer: Any, gbs: int, seq_len: int) -> BatchedDataDict:
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    generator = torch.Generator(device="cpu").manual_seed(_SEED)
    lengths = torch.tensor(
        [max(1024, seq_len * ((i % 8) + 1) // 8 // 64 * 64) for i in range(gbs)],
        dtype=torch.long,
    )
    input_ids = torch.randint(0, vocab_size, (gbs, seq_len), dtype=torch.long, generator=generator)
    token_mask = torch.zeros((gbs, seq_len), dtype=torch.float32)
    advantages = torch.zeros((gbs, seq_len), dtype=torch.float32)
    noise = torch.randn((gbs, seq_len), dtype=torch.float32, generator=generator)
    for i in range(gbs):
        prompt_len = int(lengths[i]) // 4
        token_mask[i, prompt_len : int(lengths[i])] = 1
        advantages[i, prompt_len : int(lengths[i])] = noise[i, prompt_len : int(lengths[i])]
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(gbs, dtype=torch.float32),
            "advantages": advantages,
        }
    )


def _training_warmup(policy: Any, tokenizer: Any, loss_fn: Any, master_config: Any) -> None:
    worker_group = getattr(policy, "worker_group", None)
    if worker_group is None or not hasattr(policy, "_shard_for_train"):
        _log("policy lacks the split-step worker surface; skipping training warmup")
        return
    pol_cfg = master_config.policy
    gbs = int(pol_cfg["train_global_batch_size"])
    mbs = int(pol_cfg["train_micro_batch_size"])
    seq_len = int(pol_cfg["max_total_sequence_length"])
    steps = int(os.environ.get("MLPERF_WARMUP_TRAIN_STEPS", "1"))
    batch = _synthetic_train_batch(tokenizer, gbs, seq_len)

    start = time.perf_counter()
    policy.prepare_for_lp_inference()
    try:
        logprobs = policy.get_logprobs(batch)["logprobs"]
        _log(f"logprob warmup ({gbs}x{seq_len}): {time.perf_counter() - start:.1f}s")

        # The model's own logprobs keep the PPO ratio at 1, inside the seq-mask
        # TIS window, so the backward is numerically real.
        batch["prev_logprobs"] = logprobs
        batch["generation_logprobs"] = logprobs.clone()
        batch["reference_policy_logprobs"] = logprobs.clone()

        policy.prepare_for_training()
        # Upstream split-step API: begin/train_microbatch run the production
        # fwd/bwd; abort drops the step without touching optimizer/scheduler.
        for step in range(steps):
            start = time.perf_counter()
            ray.get(
                worker_group.run_all_workers_single_data(
                    "begin_train_step", loss_fn=loss_fn, gbs=gbs, mbs=mbs
                )
            )
            try:
                futures = worker_group.run_all_workers_sharded_data(
                    "train_microbatch",
                    data=policy._shard_for_train(batch, gbs),
                    in_sharded_axes=["data_parallel"],
                    replicate_on_axes=[
                        "context_parallel",
                        "tensor_parallel",
                        "pipeline_parallel",
                    ],
                    output_is_replicated=[
                        "context_parallel",
                        "tensor_parallel",
                        "pipeline_parallel",
                    ],
                )
                worker_group.get_all_worker_results(futures)
            finally:
                ray.get(worker_group.run_all_workers_single_data("abort_train_step"))
            _log(
                f"train warmup step {step + 1}/{steps}: {time.perf_counter() - start:.1f}s "
                "(fwd+bwd complete, step aborted, optimizer untouched)"
            )
    except Exception:
        # Leave the policy in the mode the training loop expects.
        policy.prepare_for_training()
        raise


def _run_warmup_stage(name: str, fn: Callable[[], None]) -> None:
    """Run one required warmup stage and propagate failures before run_start."""
    start = time.perf_counter()
    _log(f"{name} warmup started")
    try:
        fn()
    except Exception as exc:
        _log(f"{name} warmup failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        _log(f"{name} warmup stage finished in {time.perf_counter() - start:.1f}s")


def _run_policy_generation_warmups(
    generation_fn: Callable[[], None],
    training_fn: Callable[[], None],
    *,
    generation_enabled: bool,
    training_enabled: bool,
    concurrent: bool,
) -> None:
    """Warm disjoint vLLM and policy pools concurrently when both are enabled."""
    stages = [
        ("generation", generation_enabled, generation_fn),
        ("training", training_enabled, training_fn),
    ]
    enabled_stages = [(name, fn) for name, enabled, fn in stages if enabled]
    for name, enabled, _ in stages:
        if not enabled:
            _log(f"{name} warmup disabled; skipping")

    if concurrent and len(enabled_stages) == 2:
        start = time.perf_counter()
        _log("launching generation and training warmups concurrently")
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="mlperf-warmup"
        ) as pool:
            futures = {
                pool.submit(_run_warmup_stage, name, fn): name
                for name, fn in enabled_stages
            }
            for future in as_completed(futures):
                future.result()
        _log(
            "concurrent generation and training warmups joined in "
            f"{time.perf_counter() - start:.1f}s"
        )
        return

    for name, fn in enabled_stages:
        _run_warmup_stage(name, fn)


def maybe_run_mlperf_warmup(
    policy: Any,
    policy_generation: Any,
    tokenizer: Any,
    loss_fn: Any,
    master_config: Any,
    warmup_cfg: dict[str, Any] | None = None,
) -> None:
    """Run required pre-run warmups; enabled-stage failures are fatal by design.

    Continuing after partial warmup coverage would move one-time compilation,
    autotuning, or communication initialization into the MLPerf timed region.
    Abort before init_stop/run_start instead so the reported run is not invalid.
    """
    # On by default; a config opts out with `export MLPERF_WARMUP=0`.
    if not _env_flag("MLPERF_WARMUP", "1"):
        return
    generation_cfg = master_config.policy.get("generation") or {}
    if (generation_cfg.get("colocated") or {}).get("enabled"):
        _log("colocated generation not supported; skipping warmup")
        return
    warmup_cfg = warmup_cfg or {}
    total_start = time.perf_counter()
    _log("starting untimed warmup (synthetic inputs only)")
    refit_enabled = _env_flag("MLPERF_WARMUP_REFIT") and policy_generation is not None
    if refit_enabled:
        _run_warmup_stage("refit", lambda: _dry_refit(policy, policy_generation))
    else:
        _log("refit warmup disabled; skipping")

    _run_policy_generation_warmups(
        lambda: _generation_warmup(
            policy_generation,
            tokenizer,
            master_config.policy,
            warmup_cfg,
        ),
        lambda: _training_warmup(policy, tokenizer, loss_fn, master_config),
        generation_enabled=(
            _env_flag("MLPERF_WARMUP_GEN") and policy_generation is not None
        ),
        training_enabled=_env_flag("MLPERF_WARMUP_TRAIN"),
        concurrent=_env_flag("MLPERF_WARMUP_CONCURRENT_POLICY_GENERATION"),
    )
    _log(f"warmup done in {time.perf_counter() - total_start:.1f}s")
