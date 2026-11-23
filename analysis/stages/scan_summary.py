"""
Scan summary (stage 2 of both profiles): one small table of scan-level facts per raw file.

Parses each raw file once and writes `<results>/cache/scan_summary/<sample_id>.parquet` with one row per scan
(retention time, TIC, base peak, precursor m/z). Chromatogram plotting reads these instead of re-parsing the raw
files. Needs only `context.mzml_file_paths`.

An existing parquet is reused as it is: if a raw file is replaced under the same name, delete
`cache/scan_summary/` to rebuild it.
"""
from analysis.context import LCMSContext
from analysis.utils.scan_summary import extract_scan_summary

class ScanSummaryCacheStage:
    """Write a per-scan summary parquet for every registered raw file that does not have one yet."""
    _name:str = "scan_summary_cache"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('ingestion',)

    def validate_input(self, context: LCMSContext) -> bool:
        """Needs at least one registered raw-file path."""
        return bool(context.mzml_file_paths)

    def execute(self, context: LCMSContext) -> LCMSContext:
        """Parse each raw file without a parquet; record n_processed / n_skipped (already present)."""
        out_dir = context.cache_dir / "scan_summary"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_processed, n_skipped = 0, 0
        for mzml_path in context.mzml_file_paths:
            sample_id = mzml_path.stem
            out_path = out_dir / f"{sample_id}.parquet"
            if out_path.exists():
                n_skipped += 1
                continue
            df = extract_scan_summary(mzml_path)
            df.to_parquet(out_path)
            n_processed += 1

        context.qc_metrics[f"{self._name}"] = {
            "n_processed": n_processed,
            "n_skipped": n_skipped,
        }
        context.log_step(
            self._name,
            parameters={},
            metrics={"n_processed": n_processed,
                     "n_skipped": n_skipped},
            input_files=[str(p) for p in context.mzml_file_paths],
            output_files=[str(out_dir / f"{p.stem}.parquet") for p in context.mzml_file_paths],
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The stage left its metrics on the context."""
        return self._name in context.qc_metrics