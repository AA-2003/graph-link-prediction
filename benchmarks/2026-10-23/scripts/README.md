# Benchmark analysis scripts

Run from the project root after installing the optional dependencies and completing the fixed benchmark. Raw inputs are under `reports/full-benchmark/`; curated outputs are under `benchmarks/2026-10-23/`.

Run scripts in this order:

1. `validation_probes.py`
2. `decoder_probe.py`
3. `batching_probe.py`
4. `analyze.py`
5. `numerical_audit.py`
6. `write_report.py`

For example: `python benchmarks/2026-10-23/scripts/analyze.py`. The validation probes add no test predictions. Batching retrains embeddings and can take several minutes. The main result aggregator and numerical audit require all 63 successful primary cases.
