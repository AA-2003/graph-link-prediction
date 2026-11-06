import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
CPU_METHODS = ("jaccard", "adamic-adar", "hope", "node2vec")
GNN_METHODS = ("gcn", "sage", "gat")


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "linkpred").glob("*.py")) + [ROOT / "link_prediction.py"]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def last_log_line(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 4000))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        return next((line for line in reversed(lines) if line.strip()), "")
    except OSError:
        return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the fixed three-seed link-prediction benchmark serially.")
    parser.add_argument("--output", default="reports/full-benchmark")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage", choices=("cpu", "gnn", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="GNN compute device (default: cpu).")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "datasets": ["Cora", "PubMed", "PPI"],
        "seeds": [42, 123, 2026],
        "methods": list(CPU_METHODS + GNN_METHODS),
        "epochs": 200,
        "dimensions": 64,
        "walk_length": 20,
        "num_walks": 20,
        "window": 5,
        "beta": 0.01,
        "tune": True,
        "validation_ratio": 0.1,
        "test_ratio": 0.2,
        "ppi_index": 0,
        "threads": 3,
        "hope_backend": "sparse",
        "gnn_device": args.device,
        "source_sha256": source_fingerprint(),
        "scope": "Transductive within each selected graph; PPI training graph 0, not all PPI graphs.",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            parser.error("The existing benchmark uses a different protocol or source snapshot. Choose a new output.")
    else:
        atomic_json(protocol_path, protocol)
    methods = CPU_METHODS if args.stage == "cpu" else GNN_METHODS if args.stage == "gnn" else CPU_METHODS + GNN_METHODS
    state = {
        "stage": args.stage,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "worker_python": str(Path(args.python).resolve()),
        "cases": [],
        "status": "running",
    }
    state_path = output / (args.stage + "_status.json")
    previous_cases = {}
    if state_path.exists():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_cases = {
            (case["dataset"], case["seed"], case["method"]): case for case in previous_state.get("cases", [])
        }
    total = len(protocol["datasets"]) * len(protocol["seeds"]) * len(methods)
    for dataset in protocol["datasets"]:
        for seed in protocol["seeds"]:
            for method in methods:
                case_root = output / "runs" / dataset / f"seed-{seed}" / method
                case_root.mkdir(parents=True, exist_ok=True)
                reports = sorted(case_root.glob("attempt-*/results.json"))
                completed = None
                for path in reversed(reports):
                    candidate = json.loads(path.read_text(encoding="utf-8"))
                    if candidate["metadata"].get("source_sha256") != protocol["source_sha256"]:
                        parser.error("A case has results from a different source snapshot.")
                    if candidate["results"][0]["status"] == "ok":
                        completed = path
                        break
                case = {"dataset": dataset, "seed": seed, "method": method}
                if completed is not None:
                    if not args.resume:
                        parser.error("Completed output exists; use --resume or choose a new directory.")
                    process_path = completed.parent.parent / (completed.parent.name + ".process.json")
                    previous = (
                        json.loads(process_path.read_text(encoding="utf-8"))
                        if process_path.exists()
                        else previous_cases.get((dataset, seed, method), {})
                    )
                    if previous.get("report") == str(completed.relative_to(output)):
                        case.update(previous)
                    case.update(status="ok", resumed=True, report=str(completed.relative_to(output)))
                    state["cases"].append(case)
                    atomic_json(state_path, state)
                    print(f"[{len(state['cases'])}/{total}] Reused {dataset} seed {seed} {method}", flush=True)
                    continue
                number = len(list(case_root.glob("attempt-*.log"))) + 1
                attempt = case_root / f"attempt-{number:03d}"
                log_path = case_root / f"attempt-{number:03d}.log"
                command = [
                    str(Path(args.python).resolve()),
                    str(ROOT / "link_prediction.py"),
                    "--dataset",
                    dataset,
                    "--methods",
                    method,
                    "--seed",
                    str(seed),
                    "--epochs",
                    "200",
                    "--dimensions",
                    "64",
                    "--walk-length",
                    "20",
                    "--num-walks",
                    "20",
                    "--window",
                    "5",
                    "--beta",
                    "0.01",
                    "--tune",
                    "--hope-backend",
                    "sparse",
                    "--device",
                    args.device,
                    "--threads",
                    "3",
                    "--ppi-index",
                    "0",
                    "--output",
                    str(attempt),
                    "--log-every",
                    "25",
                ]
                environment = os.environ.copy()
                environment.update(
                    PYTHONUNBUFFERED="1",
                    PYTHONHASHSEED=str(seed),
                    OMP_NUM_THREADS="3",
                    MKL_NUM_THREADS="3",
                    OPENBLAS_NUM_THREADS="3",
                    CUBLAS_WORKSPACE_CONFIG=":4096:8",
                )
                print(f"[{len(state['cases']) + 1}/{total}] Starting {dataset} seed {seed} {method}", flush=True)
                started = time.perf_counter()
                peak_rss, last_update = 0, started
                with log_path.open("w", encoding="utf-8") as stream:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    case.update(status="running", pid=process.pid, log=str(log_path.relative_to(output)))
                    state["current"] = case.copy()
                    atomic_json(state_path, state)
                    try:
                        monitored = psutil.Process(process.pid)
                        while process.poll() is None:
                            try:
                                rss = monitored.memory_info().rss
                                rss += sum(
                                    child.memory_info().rss
                                    for child in monitored.children(recursive=True)
                                    if child.is_running()
                                )
                                peak_rss = max(peak_rss, rss)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                            now = time.perf_counter()
                            if now - last_update >= 30:
                                print(
                                    f"  {dataset}/{seed}/{method}: {now - started:.0f}s; "
                                    f"peak RSS {peak_rss / 2**20:.0f} MiB; {last_log_line(log_path)}",
                                    flush=True,
                                )
                                last_update = now
                            time.sleep(1)
                    except KeyboardInterrupt:
                        process.terminate()
                        process.wait()
                        raise
                report_path = attempt / "results.json"
                case.update(
                    returncode=process.returncode,
                    wall_seconds=time.perf_counter() - started,
                    peak_process_rss_bytes=peak_rss,
                )
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    case.update(status=report["results"][0]["status"], report=str(report_path.relative_to(output)))
                else:
                    case.update(status="failed", error=last_log_line(log_path))
                atomic_json(case_root / (attempt.name + ".process.json"), case)
                state["cases"].append(case)
                state.pop("current", None)
                atomic_json(state_path, state)
                print(f"  Finished: {case['status']} in {case['wall_seconds']:.1f}s", flush=True)
    state["status"] = "complete" if all(case["status"] == "ok" for case in state["cases"]) else "has_failures"
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, state)
    print("Stage status:", state["status"], flush=True)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
