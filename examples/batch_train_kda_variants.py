#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any


@dataclass(frozen=True)
class VariantSpec:
    name: str
    train_script: str
    eval_script: str


VARIANTS: dict[str, VariantSpec] = {
    "kda": VariantSpec("kda", "examples/train_kda_wikitext103.py", "examples/eval_kda_wikitext103.py"),
    "nkda": VariantSpec("nkda", "examples/train_nkda_wikitext103.py", "examples/eval_nkda_wikitext103.py"),
    "skda": VariantSpec("skda", "examples/train_skda_wikitext103.py", "examples/eval_skda_wikitext103.py"),
    "snkda": VariantSpec("snkda", "examples/train_snkda_wikitext103.py", "examples/eval_snkda_wikitext103.py"),
    "fkda": VariantSpec("fkda", "examples/train_fkda_wikitext103.py", "examples/eval_fkda_wikitext103.py"),
    "fnkda": VariantSpec("fnkda", "examples/train_fnkda_wikitext103.py", "examples/eval_fnkda_wikitext103.py"),
    "fskda": VariantSpec("fskda", "examples/train_fskda_wikitext103.py", "examples/eval_fskda_wikitext103.py"),
    "fsnkda": VariantSpec("fsnkda", "examples/train_fsnkda_wikitext103.py", "examples/eval_fsnkda_wikitext103.py"),
}


def _parse_csv(s: str) -> list[str]:
    out: list[str] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch-train KDA variants with per-GPU job concurrency.\n"
            "This script spawns the existing per-variant WikiText-103 training scripts and writes a manifest."
        )
    )

    p.add_argument("--output_root", type=str, required=True, help="Root directory to store all runs.")
    p.add_argument("--run_name", type=str, default=None, help="Run name (default: timestamp).")

    p.add_argument(
        "--variants",
        type=str,
        default="kda,nkda,skda,snkda,fkda,fnkda,fskda,fsnkda",
        help=f"Comma-separated variants. Supported: {','.join(VARIANTS.keys())}",
    )

    p.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. '0,1,2,3'.")
    p.add_argument("--jobs_per_gpu", type=int, default=1, help="Concurrent jobs per GPU.")
    p.add_argument(
        "--keep_going",
        action="store_true",
        default=True,
        help="Deprecated (always on): continue running other jobs even if some jobs fail.",
    )

    # Dataset/tokenization
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--tokenized_cache", type=str, default="data/wt103_tok")
    p.add_argument("--num_proc", type=int, default=8)

    # Model/training hyperparams (reasonable defaults for quick iteration)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--max_train_samples", type=int, default=20000)
    p.add_argument("--max_eval_samples", type=int, default=2000)

    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--expand_v", type=float, default=1.0)
    p.add_argument("--attn_mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])
    p.add_argument("--use_short_conv", action="store_true", default=False)
    p.add_argument("--allow_neg_eigval", action="store_true", default=False)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "auto"])
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--preflight_compile", action="store_true", default=False)

    p.add_argument("--no_fuse_norm", action="store_true", default=False)
    p.add_argument("--no_fuse_swiglu", action="store_true", default=False)
    p.add_argument("--no_fuse_cross_entropy", action="store_true", default=False)

    # Cross-variant ablations (only passed to scripts that support them)
    p.add_argument("--use_qk_l2norm_in_kernel", action="store_true", default=False)
    p.add_argument("--fix_lambda", type=float, default=None)
    p.add_argument("--share_decay_gate", action="store_true", default=False)
    p.add_argument("--beta_norm_eps", type=float, default=None)

    p.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Extra args appended verbatim to every training command (shell-quoted string).",
    )

    return p.parse_args()


def _base_train_args(args: argparse.Namespace) -> list[str]:
    cmd = [
        "--tokenizer",
        args.tokenizer,
        "--dataset_name",
        args.dataset_name,
        "--dataset_config",
        args.dataset_config,
        "--text_column",
        args.text_column,
        "--tokenized_cache",
        args.tokenized_cache,
        "--seq_len",
        str(args.seq_len),
        "--num_proc",
        str(args.num_proc),
        "--max_steps",
        str(args.max_steps),
        "--max_train_samples",
        str(args.max_train_samples),
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--hidden_size",
        str(args.hidden_size),
        "--num_hidden_layers",
        str(args.num_hidden_layers),
        "--num_heads",
        str(args.num_heads),
        "--head_dim",
        str(args.head_dim),
        "--expand_v",
        str(args.expand_v),
        "--attn_mode",
        args.attn_mode,
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--weight_decay",
        str(args.weight_decay),
        "--warmup_steps",
        str(args.warmup_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--eval_steps",
        str(args.eval_steps),
        "--save_steps",
        str(args.save_steps),
        "--seed",
        str(args.seed),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
    ]
    if args.cache_dir is not None:
        cmd += ["--cache_dir", args.cache_dir]

    if args.use_short_conv:
        cmd += ["--use_short_conv"]
    if args.allow_neg_eigval:
        cmd += ["--allow_neg_eigval"]
    if args.no_fuse_norm:
        cmd += ["--no_fuse_norm"]
    if args.no_fuse_swiglu:
        cmd += ["--no_fuse_swiglu"]
    if args.no_fuse_cross_entropy:
        cmd += ["--no_fuse_cross_entropy"]

    if args.dtype == "fp16":
        cmd += ["--fp16"]
    elif args.dtype == "bf16":
        cmd += ["--bf16"]

    if args.preflight_compile:
        cmd += ["--preflight_compile"]

    return cmd


def _variant_extra_args(args: argparse.Namespace, variant: str) -> list[str]:
    cmd: list[str] = []

    # Some variants accept --use_qk_l2norm_in_kernel (kda does not).
    if args.use_qk_l2norm_in_kernel and variant in {"nkda", "skda", "snkda", "fkda", "fskda"}:
        cmd += ["--use_qk_l2norm_in_kernel"]

    if args.fix_lambda is not None and variant in {"fkda", "fnkda", "fskda", "fsnkda"}:
        cmd += ["--fix_lambda", str(args.fix_lambda)]
    if args.share_decay_gate and variant in {"fkda", "fnkda", "fskda", "fsnkda"}:
        cmd += ["--share_decay_gate"]

    if args.beta_norm_eps is not None and variant in {"snkda", "fnkda", "fsnkda"}:
        cmd += ["--beta_norm_eps", str(args.beta_norm_eps)]

    return cmd


def _build_train_cmd(args: argparse.Namespace, variant: str, out_dir: Path) -> list[str]:
    spec = VARIANTS[variant]
    cmd = [sys.executable, spec.train_script, "--output_dir", str(out_dir)]
    cmd += _base_train_args(args)
    cmd += _variant_extra_args(args, variant)
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd


def _env_for_gpu(gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


@dataclass
class Job:
    variant: str
    out_dir: Path
    cmd: list[str]
    gpu_id: str
    log_path: Path


def _run_job(job: Job, manifest_fh) -> int:
    _ensure_dir(job.out_dir)
    _ensure_dir(job.log_path.parent)
    record: dict[str, Any] = {
        "variant": job.variant,
        "output_dir": str(job.out_dir),
        "eval_script": VARIANTS[job.variant].eval_script,
        "gpu_id": job.gpu_id,
        "cmd": job.cmd,
        "log": str(job.log_path),
        "time_start": time.time(),
    }
    manifest_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_fh.flush()

    print(f"[train:{job.variant}] gpu={job.gpu_id} out={job.out_dir}", flush=True)
    with open(job.log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in job.cmd) + "\n\n")
        f.flush()
        proc = None
        try:
            import subprocess

            proc = subprocess.run(job.cmd, env=_env_for_gpu(job.gpu_id), stdout=f, stderr=subprocess.STDOUT)
            rc = int(proc.returncode)
        except Exception as e:
            f.write(f"\n[runner-error] {e}\n")
            rc = 1

    return rc


def main() -> None:
    args = parse_args()
    run_name = args.run_name or _timestamp()
    output_root = Path(args.output_root).resolve()
    run_root = output_root / run_name
    logs_dir = run_root / "logs"
    _ensure_dir(run_root)
    _ensure_dir(logs_dir)

    variants = _parse_csv(args.variants)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Supported: {sorted(VARIANTS)}")

    gpus = _parse_csv(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified.")
    if args.jobs_per_gpu <= 0:
        raise SystemExit("--jobs_per_gpu must be > 0")

    workers: list[tuple[str, int]] = []
    for gid in gpus:
        for slot in range(args.jobs_per_gpu):
            workers.append((gid, slot))

    jobs: list[Job] = []
    for v in variants:
        out_dir = run_root / v
        log_path = logs_dir / f"{v}.train.log"
        cmd = _build_train_cmd(args, v, out_dir)
        jobs.append(Job(variant=v, out_dir=out_dir, cmd=cmd, gpu_id="", log_path=log_path))

    q: Queue[Job] = Queue()
    for j in jobs:
        q.put(j)

    lock = threading.Lock()
    failures: list[tuple[str, int]] = []
    manifest_path = run_root / "runs.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as manifest_fh:

        def worker_fn(gpu_id: str, slot: int) -> None:
            while True:
                try:
                    job = q.get_nowait()
                except Exception:
                    return
                job.gpu_id = gpu_id
                rc = _run_job(job, manifest_fh)
                with lock:
                    if rc != 0:
                        failures.append((job.variant, rc))
                q.task_done()

                # Always keep going: failures are recorded and reported at the end.

        threads: list[threading.Thread] = []
        for gpu_id, slot in workers:
            t = threading.Thread(target=worker_fn, args=(gpu_id, slot), daemon=True)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    if failures:
        print(f"[done] failures={failures} (see {manifest_path})", file=sys.stderr)
        raise SystemExit(1)

    print(f"[done] all ok. manifest={manifest_path}")


if __name__ == "__main__":
    main()
