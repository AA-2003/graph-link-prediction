import argparse
from datetime import datetime, timezone

from linkpred.experiment import METHODS, RunConfig, run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare graph link prediction methods on shared edge splits.")
    parser.add_argument("--dataset", choices=("karate", "Cora", "PubMed", "PPI"), default="karate")
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, "all"), default=["jaccard", "adamic-adar"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--hope-max-nodes", type=int, default=4000)
    parser.add_argument("--hope-backend", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--tune", action="store_true", help="Compare two embedding configurations using validation scores."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--ppi-index", type=int, default=0)
    parser.add_argument("--cache", default="data")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true", help="Use offline karate data and short training settings.")
    args = parser.parse_args(argv)
    if "all" in args.methods and len(args.methods) != 1:
        parser.error("Use --methods all by itself.")
    methods = METHODS if args.methods == ["all"] else args.methods
    config = RunConfig(
        dataset="karate" if args.dry_run else args.dataset,
        seed=args.seed,
        val_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        dimensions=8 if args.dry_run else args.dimensions,
        epochs=2 if args.dry_run else args.epochs,
        beta=args.beta,
        max_hope_nodes=args.hope_max_nodes,
        walk_length=5 if args.dry_run else args.walk_length,
        num_walks=3 if args.dry_run else args.num_walks,
        window=args.window,
        tune=args.tune,
        device=args.device,
        threads=args.threads,
        ppi_index=args.ppi_index,
        hope_backend=args.hope_backend,
        log_every=args.log_every,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or f"reports/{config.dataset}-{timestamp}"
    try:
        report = run_experiment(config, methods, output, args.cache)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")
    for row in report["results"]:
        if row["status"] == "ok":
            print(f"{row['method']}: AUC={row['test']['auc_roc']:.4f}, AP={row['test']['average_precision']:.4f}")
    print(f"Saved results to {output}")
    return int(any(row["status"] == "error" for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
