#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue


LOSS_RE = re.compile(r"loss=(?P<loss>[0-9.]+)\\s+ppl=(?P<ppl>[0-9.]+)")


def _parse_csv(s: str) -> list[str]:
    out: list[str] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-evaluate KDA variants from a training manifest (runs.jsonl).")
    p.add_argument("--manifest", type=str, required=True, help="Path to runs.jsonl produced by batch_train script.")
    p.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. '0,1'.")
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--keep_going", action="store_true")

    p.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "auto"])

    p.add_argument("--generate", action="store_true", default=False)
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--max_new_tokens", type=int, default=64)

    p.add_argument("--tokenized_cache", type=str, default=None, help="Override tokenized cache used for eval.")
    p.add_argument("--extra_args", type=str, default="", help="Extra args appended verbatim to every eval command.")

    return p.parse_args()


def _env_for_gpu(gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _dtype_flag(dtype: str) -> str:
    if dtype not in {"fp16", "bf16", "fp32", "auto"}:
        raise ValueError(dtype)
    return dtype


@dataclass
class EvalJob:
    variant: str
    model_dir: Path
    eval_script: str
    gpu_id: str
    log_path: Path


def _build_eval_cmd(args: argparse.Namespace, job: EvalJob) -> list[str]:
    cmd = [
        sys.executable,
        job.eval_script,
        "--model",
        str(job.model_dir),
        "--split",
        args.split,
        "--seq_len",
        str(args.seq_len),
        "--max_samples",
        str(args.max_samples),
        "--batch_size",
        str(args.batch_size),
        "--dtype",
        _dtype_flag(args.dtype),
    ]
    if args.tokenized_cache is not None:
        cmd += ["--tokenized_cache", args.tokenized_cache]
    if args.generate:
        cmd += ["--generate", "--prompt", args.prompt, "--max_new_tokens", str(args.max_new_tokens)]
    if args.extra_args:
        cmd += shlex.split(args.extra_args)
    return cmd


def _parse_metrics(log_text: str) -> tuple[float | None, float | None]:
    m = LOSS_RE.search(log_text)
    if not m:
        return None, None
    return float(m.group("loss")), float(m.group("ppl"))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    gpus = _parse_csv(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified.")
    if args.jobs_per_gpu <= 0:
        raise SystemExit("--jobs_per_gpu must be > 0")

    workers: list[tuple[str, int]] = []
    for gid in gpus:
        for slot in range(args.jobs_per_gpu):
            workers.append((gid, slot))

    run_root = manifest_path.parent
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_root / f"eval_{args.split}_{_timestamp()}.csv"

    # Read manifest and build jobs.
    jobs: list[EvalJob] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            variant = str(rec["variant"])
            model_dir = Path(rec["output_dir"])
            eval_script = str(rec.get("eval_script") or "")  # backward compat
            if not eval_script:
                # Heuristic: map from training script name.
                train_script = str(rec["cmd"][1])
                eval_script = train_script.replace("train_", "eval_")
            log_path = logs_dir / f"{variant}.eval.log"
            jobs.append(EvalJob(variant=variant, model_dir=model_dir, eval_script=eval_script, gpu_id="", log_path=log_path))

    q: Queue[EvalJob] = Queue()
    for j in jobs:
        q.put(j)

    lock = threading.Lock()
    failures: list[str] = []
    results: dict[str, dict[str, object]] = {}

    def run_one(job: EvalJob) -> int:
        cmd = _build_eval_cmd(args, job)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[eval:{job.variant}] gpu={job.gpu_id} model={job.model_dir}", flush=True)
        import subprocess

        with open(job.log_path, "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
            f.flush()
            proc = subprocess.run(cmd, env=_env_for_gpu(job.gpu_id), stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            return int(proc.returncode)
        txt = job.log_path.read_text(encoding="utf-8", errors="ignore")
        loss, ppl = _parse_metrics(txt)
        with lock:
            results[job.variant] = {
                "variant": job.variant,
                "model_dir": str(job.model_dir),
                "loss": loss,
                "ppl": ppl,
                "log": str(job.log_path),
            }
        return 0

    def worker_fn(gpu_id: str, slot: int) -> None:
        while True:
            try:
                job = q.get_nowait()
            except Exception:
                return
            job.gpu_id = gpu_id
            rc = run_one(job)
            with lock:
                if rc != 0:
                    failures.append(job.variant)
            q.task_done()
            if rc != 0 and not args.keep_going:
                while True:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except Exception:
                        break
                return

    threads: list[threading.Thread] = []
    for gpu_id, slot in workers:
        t = threading.Thread(target=worker_fn, args=(gpu_id, slot), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Write results CSV.
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "model_dir", "loss", "ppl", "log"])
        w.writeheader()
        for v in sorted(results):
            w.writerow(results[v])

    if failures:
        print(f"[done] failures={failures} results={results_path}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[done] all ok. results={results_path}")


if __name__ == "__main__":
    main()

