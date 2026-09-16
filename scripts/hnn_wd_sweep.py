"""Control experiment for Experiment 7 (docs/hnn_dynamics.md): does plain global
AdamW weight_decay on the HNN reach the same grokking speed/quality as the scoped
post-step shrinkage mechanism (`shrink_target`), or is scoping to the output-layer
weight actually doing the work?

Exp 7 arm 0 (`output_bias=True, fixed_x=True, shrink_target="both", shrink_wd=4.0`)
memorised@4k, hit test>=90%@24k, peaked at 98.1%. This sweep reruns the *identical*
setup but swaps the scoped, Adam-bypassing shrinkage for plain `weight_decay` (optax
AdamW's own decoupled WD, applied to every HNN parameter — W0/b0 and W1/b_last alike
— not just the output layer). fixed_x=True is held constant so this isolates exactly
one variable: scoped-and-bypassing vs. global-and-Adam-coupled delivery of the same
nominal decay magnitude.

Hypothesis: global weight_decay will NOT match arm 0, because it's one coefficient
doing two jobs with opposite needs — shrinking the carrier (W1/b_last) enough to reach
the ~190-norm equilibrium while also shrinking the hidden layer (W0/b0) that needs to
stay expressive. Historical data point (Exp 3, random x): wd_hnn=0.5 prevented
memorisation/consolidation entirely rather than producing the grokking ratchet. This
sweep checks whether that holds under fixed_x too, and whether any single coefficient
threads the needle.

shrink_wd=4.0 and weight_decay=4.0 are the *same* nominal per-step decay magnitude
(both are `optax.adamw`-style decoupled shrinkage: param *= (1 - lr * coefficient)) —
that value anchors the sweep grid, bracketed on both sides.

Runs the grid in parallel across CUDA GPUs — one subprocess per weight_decay value,
each pinned to a single GPU via CUDA_VISIBLE_DEVICES (same pattern as
hnn_grid_search.py). Each run is tiny (85k params, full-batch) and independent, so
this parallelises trivially across trials rather than within one.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/hnn_wd_sweep.py
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/hnn_wd_sweep.py --weight-decays 4.0 8.0 32.0
    uv run scripts/hnn_wd_sweep.py --dry-run

Produces:
    <output-dir>/wd<value>/config.json   — one directory per weight_decay value
    <output-dir>/wd<value>/metrics.json
    <output-dir>/summary.csv             — aggregated results at end
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
from pathlib import Path
from typing import Any

DEFAULT_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]


def _get_gpu_slots() -> list[int]:
    """Return GPU IDs from CUDA_VISIBLE_DEVICES. Fails loudly if not set."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is not set.\n"
            "  e.g.  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run scripts/hnn_wd_sweep.py"
        )
    if cvd in ("", "NoDevices", "-1"):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={cvd!r} — no GPUs available.")
    return [int(x.strip()) for x in cvd.split(",")]


def _run_id(wd: float) -> str:
    return f"wd{wd:g}"


def _build_configs(output_dir: Path, args: argparse.Namespace) -> list[dict]:
    configs = []
    for wd in args.weight_decays:
        run_id = _run_id(wd)
        configs.append({
            "data_dir": str(output_dir / run_id),
            "modulus": 97,
            "target_hidden": 32,
            "target_depth": 1,
            "n_stimulus": 8,
            "stim_ffn_hidden": 8,
            "stim_ffn_depth": 1,
            "init_scale": 0.1,
            "output_bias": True,
            "train_fraction": args.train_fraction,
            "lr": 1e-3,
            "weight_decay": wd,        # the variable under test — plain global AdamW WD
            "grad_clip": 1.0,
            "lambda_complexity": 0.0,  # isolate weight_decay — no softmin-diversity confound
            "shrink_target": "none",   # isolate weight_decay — no scoped-shrink confound
            "fixed_x": True,           # matches Exp 7 arm 0's structural setup exactly
            "n_mc": 1,
            "n_epochs": args.n_epochs,
            "log_interval": args.log_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "hexplot_interval": 0,
            "seed": args.seed,
        })
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
    import json, sys
    from god.training.grokking_hnn import GrokkingHNNConfig, train
    with open(sys.argv[1]) as f:
        d = json.load(f)
    cfg = GrokkingHNNConfig(**d)
    train(cfg)
""")


def _run_job(cfg: dict, gpu_id: int, dry_run: bool) -> dict:
    """Run a single weight_decay config in a subprocess pinned to gpu_id."""
    run_dir = Path(cfg["data_dir"])
    run_id = run_dir.name

    if _is_complete(cfg):
        print(f"[SKIP] {run_id} (already complete)")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        return {"run_id": run_id, "status": "skipped", **_summarise(cfg, metrics)}

    print(f"[GPU {gpu_id}] START {run_id}")
    if dry_run:
        return {"run_id": run_id, "status": "dry_run"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
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


def _summarise(cfg: dict, metrics: list[dict[str, Any]]) -> dict:
    """Extract key summary stats from a completed run's metrics."""
    if not metrics:
        return {}
    final = metrics[-1]
    peak_test = max(m["test_acc"] for m in metrics)
    mem_epoch = next((m["epoch"] for m in metrics if m["train_acc"] >= 0.99), None)
    grok_epoch = next((m["epoch"] for m in metrics if m["test_acc"] >= 0.90), None)
    return {
        "weight_decay": cfg["weight_decay"],
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "peak_test_acc": peak_test,
        "memorised_epoch": mem_epoch,
        "grokked_epoch": grok_epoch,
        "final_param_norm": final.get("param_norm"),
        "final_probe_bias_norm": final.get("probe_bias_norm"),
        "final_probe_residual_norm": final.get("probe_residual_norm"),
    }


def _write_summary(results: list[dict], output_dir: Path) -> None:
    rows = [r for r in results if r.get("status") not in ("dry_run",)]
    if not rows:
        return
    fields = [
        "run_id", "status", "weight_decay",
        "final_train_acc", "final_test_acc", "peak_test_acc",
        "memorised_epoch", "grokked_epoch",
        "final_param_norm", "final_probe_bias_norm", "final_probe_residual_norm",
    ]
    out = output_dir / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r.get("weight_decay", 0)))
    print(f"\nSummary → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/hnn_wd_sweep")
    parser.add_argument("--weight-decays", type=float, nargs="+", default=DEFAULT_GRID)
    parser.add_argument("--jobs-per-gpu", type=int, default=1, metavar="N",
                         help="Concurrent jobs per GPU (use >1 if jobs fit in memory)")
    parser.add_argument("--n-epochs", type=int, default=200_000)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=2000)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs(output_dir, args)
    n_complete = sum(1 for c in configs if _is_complete(c))
    print(f"Grid: {len(configs)} weight_decay values: {args.weight_decays}")
    print(f"Already complete: {n_complete}  |  Pending: {len(configs) - n_complete}")

    gpu_ids = _get_gpu_slots()
    gpu_slots = [g for g in gpu_ids for _ in range(args.jobs_per_gpu)]
    n_workers = len(gpu_slots)
    print(f"GPUs: {gpu_ids}")
    print(f"Workers: {n_workers} ({len(gpu_ids)} GPU(s) × {args.jobs_per_gpu} job(s)/GPU)")

    if args.dry_run:
        for cfg in configs:
            run_id = Path(cfg["data_dir"]).name
            done = "[done]" if _is_complete(cfg) else "[run] "
            print(f"  {done} {run_id}  (weight_decay={cfg['weight_decay']})")
        return

    results = []
    futures = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, cfg in enumerate(configs):
            gpu_id = gpu_slots[i % len(gpu_slots)]
            future = pool.submit(_run_job, cfg, gpu_id, False)
            futures[future] = cfg

        for future in as_completed(futures):
            try:
                results.append(future.result())
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
