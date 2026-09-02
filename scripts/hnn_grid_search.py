"""Grid search over lambda_complexity × softmin_temp × softmax_temp for HNN grokking.

Runs the full grid in parallel across CUDA GPUs. One subprocess per job, each
pinned to a single GPU via CUDA_VISIBLE_DEVICES. XLA_PYTHON_CLIENT_PREALLOCATE=false
lets multiple jobs share GPU memory without OOM.

Usage:
    uv run scripts/hnn_grid_search.py --gpus 4 --jobs-per-gpu 2 --output-dir data/grid
    uv run scripts/hnn_grid_search.py --gpus 1 --output-dir data/grid --dry-run

Produces:
    <output-dir>/<run-id>/config.json   — one directory per config
    <output-dir>/<run-id>/metrics.json
    <output-dir>/summary.csv            — aggregated results at end
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------

LAMBDA_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SOFTMIN_TEMP_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SOFTMAX_TEMP_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

# Diagonal: lam=1e-2 configs where smax > smin and peak_test_acc > 0.01
DIAGONAL_CONFIGS = [
    (1e-2, 0.01, 1.0),
    (1e-2, 0.01, 10.0),
    (1e-2, 0.01, 100.0),
    (1e-2, 0.1,  1.0),
    (1e-2, 0.1,  10.0),
    (1e-2, 0.1,  100.0),
    (1e-2, 1.0,  10.0),
    (1e-2, 1.0,  100.0),
]


def _get_gpu_slots() -> list[int]:
    """Return GPU IDs from CUDA_VISIBLE_DEVICES. Fails loudly if not set."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is not set.\n"
            "  e.g.  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/hnn_grid_search.py"
        )
    if cvd in ("", "NoDevices", "-1"):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={cvd!r} — no GPUs available.")
    return [int(x.strip()) for x in cvd.split(",")]


def _run_id(lam: float, softmin: float, softmax: float, n_mc: int) -> str:
    return f"lam{lam:.0e}_smin{softmin:.0e}_smax{softmax:.0e}_mc{n_mc}"


def _build_configs(
    output_dir: Path,
    n_epochs: int,
    n_mc_values: list[int],
    seed: int,
    diagonal: bool,
) -> list[dict]:
    grid = DIAGONAL_CONFIGS if diagonal else list(product(LAMBDA_GRID, SOFTMIN_TEMP_GRID, SOFTMAX_TEMP_GRID))
    configs = []
    for lam, softmin, softmax in grid:
        for n_mc in n_mc_values:
            run_id = _run_id(lam, softmin, softmax, n_mc)
            cfg = {
                "data_dir": str(output_dir / run_id),
                "modulus": 97,
                "target_hidden": 32,
                "target_depth": 1,
                "n_stimulus": 8,
                "stim_ffn_hidden": 8,
                "stim_ffn_depth": 1,
                "init_scale": 0.1,
                "train_fraction": 0.3,
                "n_epochs": n_epochs,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "lambda_complexity": lam,
                "softmax_temp": softmax,
                "softmin_temp": softmin,
                "lambda_target_decay": 0.0,
                "n_mc": n_mc,
                "fixed_x": False,
                "log_interval": max(n_epochs // 50, 1000),
                "hexplot_interval": 0,
                "checkpoint_interval": 0,
                "hexplot_xlim": 1000.0,
                "hexplot_ylim": 5.0,
                "seed": seed,
            }
            configs.append(cfg)
    return configs


def _is_complete(cfg: dict) -> bool:
    """True if the run has already finished (metrics.json covers full training)."""
    metrics_path = Path(cfg["data_dir"]) / "metrics.json"
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
        return len(metrics) > 0 and metrics[-1]["epoch"] >= cfg["n_epochs"]
    except (json.JSONDecodeError, KeyError):
        return False


_RUNNER = textwrap.dedent("""\
    import json, sys, dataclasses
    from god.training.grokking_hnn import GrokkingHNNConfig, train
    with open(sys.argv[1]) as f:
        d = json.load(f)
    cfg = GrokkingHNNConfig(**d)
    train(cfg)
""")


def _run_job(cfg: dict, gpu_id: int, dry_run: bool) -> dict:
    """Run a single grid search config in a subprocess pinned to gpu_id."""
    run_dir = Path(cfg["data_dir"])
    run_id = run_dir.name

    if _is_complete(cfg):
        print(f"[SKIP] {run_id} (already complete)")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        return {"run_id": run_id, "status": "skipped", **_summarise(cfg, metrics)}

    print(f"[GPU {gpu_id}] START {run_id}")
    if dry_run:
        return {"run_id": run_id, "status": "dry_run"}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(cfg, f)
        cfg_path = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(_RUNNER)
        runner_path = f.name

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        log_path = run_dir / "stdout.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            result = subprocess.run(
                [sys.executable, runner_path, cfg_path],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        if result.returncode != 0:
            print(f"[GPU {gpu_id}] FAILED {run_id} (exit {result.returncode})")
            return {"run_id": run_id, "status": "failed", "exit_code": result.returncode}

        metrics = json.loads((run_dir / "metrics.json").read_text())
        print(f"[GPU {gpu_id}] DONE  {run_id}")
        return {"run_id": run_id, "status": "ok", **_summarise(cfg, metrics)}

    finally:
        os.unlink(cfg_path)
        os.unlink(runner_path)


def _summarise(cfg: dict, metrics: list[dict]) -> dict:
    """Extract key summary stats from a completed run's metrics."""
    if not metrics:
        return {}
    final = metrics[-1]
    peak_test = max(m["test_acc"] for m in metrics)
    mem_epoch = next(
        (m["epoch"] for m in metrics if m["train_acc"] >= 0.99), None
    )
    grok_epoch = next(
        (m["epoch"] for m in metrics if m["test_acc"] >= 0.90), None
    )
    return {
        "lambda_complexity": cfg["lambda_complexity"],
        "softmin_temp": cfg["softmin_temp"],
        "softmax_temp": cfg["softmax_temp"],
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "peak_test_acc": peak_test,
        "memorised_epoch": mem_epoch,
        "grokked_epoch": grok_epoch,
        "final_param_norm": final.get("param_norm"),
        "final_param_std": final.get("param_std"),
    }


def _write_summary(results: list[dict], output_dir: Path) -> None:
    rows = [r for r in results if r.get("status") not in ("dry_run",)]
    if not rows:
        return
    fields = [
        "run_id", "status",
        "lambda_complexity", "softmin_temp", "softmax_temp",
        "final_train_acc", "final_test_acc", "peak_test_acc",
        "memorised_epoch", "grokked_epoch",
        "final_param_norm", "final_param_std",
    ]
    out = output_dir / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HNN grid search over lambda × softmin_temp × softmax_temp")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, metavar="N",
                        help="Concurrent jobs per GPU (use >1 if jobs fit in memory)")
    parser.add_argument("--output-dir", default="data/hnn_grid", help="Root output directory")
    parser.add_argument("--n-epochs", type=int, default=100_000, help="Epochs per run")
    parser.add_argument("--n-mc", type=int, nargs="+", default=[4], metavar="N",
                        help="MC noise samples per step (multiple values run as separate configs)")
    parser.add_argument("--diagonal", action="store_true",
                        help="Use only the diagonal lam=1e-2 configs (smax > smin) instead of full grid")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs(output_dir, args.n_epochs, args.n_mc, args.seed, args.diagonal)
    total = len(configs)
    if args.diagonal:
        print(f"Grid: {len(DIAGONAL_CONFIGS)} diagonal configs × {len(args.n_mc)} n_mc values = {total} configs")
    else:
        print(f"Grid: {len(LAMBDA_GRID)} λ × {len(SOFTMIN_TEMP_GRID)} softmin × {len(SOFTMAX_TEMP_GRID)} softmax × {len(args.n_mc)} n_mc = {total} configs")

    n_complete = sum(1 for c in configs if _is_complete(c))
    n_pending = total - n_complete
    print(f"Already complete: {n_complete}  |  Pending: {n_pending}")

    gpu_ids = _get_gpu_slots()
    gpu_slots = [g for g in gpu_ids for _ in range(args.jobs_per_gpu)]
    n_workers = len(gpu_slots)
    print(f"GPUs: {gpu_ids}")
    print(f"Workers: {n_workers} ({len(gpu_ids)} GPU(s) × {args.jobs_per_gpu} job(s)/GPU)")

    if args.dry_run:
        for cfg in configs:
            run_id = Path(cfg["data_dir"]).name
            done = "[done]" if _is_complete(cfg) else "[run] "
            print(f"  {done} {run_id}")
        return

    results = []
    slot_index = 0
    futures = {}

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for cfg in configs:
            gpu_id = gpu_slots[slot_index % len(gpu_slots)]
            slot_index += 1
            future = pool.submit(_run_job, cfg, gpu_id, False)
            futures[future] = cfg

        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
            except Exception as e:
                cfg = futures[future]
                run_id = Path(cfg["data_dir"]).name
                print(f"[ERROR] {run_id}: {e}")
                results.append({"run_id": run_id, "status": "error", "error": str(e)})

    _write_summary(results, output_dir)

    ok = sum(1 for r in results if r.get("status") in ("ok", "skipped"))
    failed = sum(1 for r in results if r.get("status") in ("failed", "error"))
    print(f"\nDone. {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
