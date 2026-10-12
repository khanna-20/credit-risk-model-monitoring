"""Command-line entry point.

    python -m creditrisk validate   # data-quality checks only
    python -m creditrisk develop    # validate, train, evaluate, save model + reports
    python -m creditrisk simulate   # write SIMULATED monitoring batches
    python -m creditrisk monitor    # monitor every batch CSV in the batch folder
    python -m creditrisk all        # develop -> simulate -> monitor
"""
from __future__ import annotations

import argparse
import sys
import time

import joblib

from .config import load_config, output_dirs
from .data import load_dataset
from .validation import run_data_quality_checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m creditrisk", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["validate", "develop", "simulate", "monitor", "all"])
    parser.add_argument("--config", default=None, help="path to a YAML config (default: config/config.yaml)")
    parser.add_argument("--batch-dir", default=None, help="folder of batch CSVs to monitor (monitor only)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    start = time.time()

    if args.command == "validate":
        cfg_d = cfg["data"]
        dq = run_data_quality_checks(load_dataset(cfg), cfg_d.get("expected_rows"), cfg_d.get("expected_defaults"),
                                     cfg["features"]["max_missing_rate"])
        print(dq.to_string(index=False))
        out = output_dirs(cfg)["reports"] / "data_quality_checks.csv"
        dq.to_csv(out, index=False)
        return 1 if (dq["status"] == "FAIL").any() else 0

    from .monitoring import run_monitoring
    from .simulate import simulate_batches
    from .train import develop

    if args.command in ("develop", "all"):
        develop(cfg)
    if args.command in ("simulate", "all"):
        bundle = joblib.load(output_dirs(cfg)["model"] / "model_bundle.joblib")
        simulate_batches(cfg, bundle)
    if args.command in ("monitor", "all"):
        run_monitoring(cfg, args.batch_dir)
    print(f"[done] {args.command} finished in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
