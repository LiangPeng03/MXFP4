#!/usr/bin/env python3
"""Run the remaining MXFP4 comparisons on GPU 0 and 1 and resume from the CSV."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


MODELS = (
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen3-1.7B",
    "meta-llama/Meta-Llama-3-8B",
    "Qwen/Qwen3-8B",
)
METHODS = ("BF16", "GPTQ", "MR-GPTQ")
GPUS = (0, 1)
TASKS = (
    "winogrande", "hellaswag", "piqa", "arc_challenge", "boolq",
    "gsm8k", "mmlu", "arc_easy", "mbpp", "lambada_openai",
)
METRICS = ("wikitext2_ppl", "c4_ppl", *TASKS)
FIELDS = (
    "model", "method", "w_bits", "a_bits", "transform", "hadamard_size",
    "seed", "lm_eval_seed", "gpu", "status", "source", "seconds", *METRICS, "error",
)
SMOL = MODELS[0]
SUMMARY_PREFIX = "[RESULT_SUMMARY] "
PREVIOUS = {
    ("BF16", SMOL): (
        "previous_summary",
        (15.431035995483398, 21.35453224182129, 0.5303867403314917, 0.4495,
         0.6838955386289445, 0.32849829351535836, 0.6039755351681957,
         0.009097801364670205, 0.2553631710769901, 0.6738215488215489,
         0.034, 0.4281001358431981),
    ),
    ("GPTQ", SMOL): (
        "screenshot_rounded",
        (43.8384, 60.199, 0.5051, 0.3690, 0.6023, 0.2628, 0.5538,
         0.0106, 0.2594, 0.4928, 0.0, 0.1822),
    ),
}


def init_rows(path):
    rows = {}
    for model in MODELS:
        for method in METHODS:
            key = (method, model)
            bits = 16 if method == "BF16" else 4
            group = ""
            if method == "MR-GPTQ":
                group = 64 if model == SMOL else 128
            row = dict(
                model=model, method=method, w_bits=bits, a_bits=bits,
                transform="hadamard" if group else "identity", hadamard_size=group,
                seed=0, lm_eval_seed=0, status="pending",
            )
            if key in PREVIOUS:
                source, scores = PREVIOUS[key]
                row.update(dict(zip(METRICS, scores, strict=True)), status="complete", source=source)
            rows[key] = row

    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"Unexpected CSV columns in {path}. Choose a different --out path.")
            seen = set()
            for row in reader:
                key = (row["method"], row["model"])
                if key not in rows or key in seen:
                    raise ValueError(f"Unexpected or duplicate experiment in {path}: {key}")
                seen.add(key)
                expected = rows[key]
                if row["hadamard_size"] != str(expected["hadamard_size"]):
                    continue
                if row["status"] == "complete" or (
                    row["status"] == "failed" and expected["status"] == "pending"
                ):
                    rows[key] = row
    return rows


def write_csv(path, rows):
    file = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    )
    temporary = Path(file.name)
    try:
        with file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows.values())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_env(gpu):
    return dict(
        os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHON_BIN=sys.executable, PYTHONUNBUFFERED="1",
    )


def preflight(repo):
    subprocess.run(
        ["bash", "run_test.sh", "--help"], cwd=repo, env=gpu_env(GPUS[0]),
        stdout=subprocess.DEVNULL, check=True,
    )
    check = (
        "import torch; assert torch.cuda.device_count() == 1, 'Expected one visible CUDA GPU'; "
        "props = torch.cuda.get_device_properties(0); "
        "torch.empty(1, device='cuda'); print(f'{props.name}, {props.total_memory / 1024**3:.1f} GiB')"
    )
    for gpu in GPUS:
        device = subprocess.check_output(
            [sys.executable, "-c", check], env=gpu_env(gpu), text=True
        ).strip()
        print(f"GPU {gpu}: {device}", flush=True)


def parse_scores(line, expected_model):
    summary = json.loads(line.removeprefix(SUMMARY_PREFIX))
    if summary["model"] != expected_model:
        raise ValueError(f"Expected {expected_model}, got {summary['model']}")
    scores = {name: summary[name] for name in METRICS[:2]}
    scores.update({task: summary["results"][task] for task in TASKS})
    for name, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid {name}: {value!r}")
    return scores


def run_job(repo, row, gpu):
    label = f"GPU {gpu} | {row['model']} | {row['method']}"
    print(f"START  {label}", flush=True)
    start = time.monotonic()
    record = dict(row, gpu=gpu, status="failed", source="run")
    command = [
        "bash", "run_test.sh", "--w_bits", str(row["w_bits"]), "--a_bits", str(row["a_bits"]),
        "--seed", "0", "--lm_eval_seed", "0", "--lm_eval_tasks", *TASKS,
    ]
    if row["method"] != "BF16":
        command += [
            "--gptq", "--quantization_order", "activation", "--transform_class", row["transform"],
            "--w_observer", "mse", "--a_observer", "minmax", "--num_sequences", "128",
            "--rel_damp", "0.01",
        ]
    if row["hadamard_size"]:
        command += ["--hadamard_group_size", str(row["hadamard_size"])]

    tail = deque(maxlen=12)
    summary_line = None
    try:
        env = dict(gpu_env(gpu), MODEL=row["model"])
        with subprocess.Popen(
            command, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        ) as process:
            for line in process.stdout:
                line = line.strip()
                if line.startswith(SUMMARY_PREFIX):
                    summary_line = line
                elif line:
                    tail.append(line[-500:])
            exit_code = process.wait()
        if exit_code:
            raise ValueError(f"exit {exit_code}: {tail[-1] if tail else 'no process output'}")
        if summary_line is None:
            raise ValueError("Missing [RESULT_SUMMARY] output")
        record.update(parse_scores(summary_line, row["model"]), status="complete", error="")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record["error"] = str(exc)
        error_tail = "\n".join(tail)
        print(f"FAILED {label} | {exc}\n{error_tail}", flush=True)
    record["seconds"] = round(time.monotonic() - start, 1)
    return record


def run_queue(repo, path, rows):
    # Start with the two 8B models, then pair the next two sizes, and keep both GPUs busy.
    order = (
        (method, model)
        for pair_start in (4, 2, 0)
        for method in METHODS
        for model in MODELS[pair_start:pair_start + 2]
    )
    pending = [key for key in order if rows[key]["status"] in {"pending", "failed"}]
    print(f"Queued {len(pending)} experiments. Results: {path}", flush=True)
    if not pending:
        return
    preflight(repo)
    queue = iter(pending)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        running = {}
        for gpu in GPUS:
            key = next(queue, None)
            if key is not None:
                running[pool.submit(run_job, repo, rows[key], gpu)] = (gpu, key)
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                gpu, key = running.pop(future)
                row = future.result()
                rows[key] = row
                write_csv(path, rows)
                if row["status"] == "complete":
                    print(f"DONE   GPU {gpu} | {row['model']} | {row['method']} | {row['seconds']} s", flush=True)
                key = next(queue, None)
                if key is not None:
                    running[pool.submit(run_job, repo, rows[key], gpu)] = (gpu, key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="FP-Quant directory containing run_test.sh")
    parser.add_argument("--out", type=Path, default=Path("mxfp4_results.csv"), help="CSV to create or resume")
    args = parser.parse_args()
    repo = args.repo.resolve()
    path = args.out.resolve()
    if not (repo / "run_test.sh").is_file():
        parser.error(f"run_test.sh not found in {repo}. Specify the FP-Quant directory with --repo.")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rows = init_rows(path)
        write_csv(path, rows)
        run_queue(repo, path, rows)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Batch stopped: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"Interrupted. Resume from {path} with the same command.", file=sys.stderr)
        return 130
    complete = sum(row["status"] == "complete" for row in rows.values())
    failed = sum(row["status"] == "failed" for row in rows.values())
    print(f"Complete: {complete}, failed: {failed}. CSV: {path}")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
