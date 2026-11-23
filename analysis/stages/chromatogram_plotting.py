"""
Chromatogram plotting (stage 3 of both profiles): TIC and BPC plots per sample, plus cohort overlays.

Reads the scan-summary parquet files (no raw-file parsing) and writes to `<results>/plots/chromatograms/`:
`<sample_id>_tic.png`, `<sample_id>_bpc.png`, `overlay_tic.png`, `overlay_bpc.png`. The overlays colour each
sample by its `sample_role`.

It plots every parquet found in `cache/scan_summary/`, including leftovers of samples that were removed from the
study since; clear that folder if the sample list shrinks.
"""
import pandas as pd

from analysis.context import LCMSContext
from analysis.utils.plotting import plot_tic, plot_bpc, plot_tic_overlay, plot_bpc_overlay


class ChromatogramPlottingStage:
    """Render per-sample and overlay TIC/BPC figures from the cached scan summaries."""
    _name: str = "chromatogram_plotting"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('scan_summary_cache',)

    def validate_input(self, context: LCMSContext) -> bool:
        """Needs at least one scan-summary parquet from the previous stage."""
        cache_dir = context.cache_dir / "scan_summary"
        return cache_dir.exists() and any(cache_dir.glob("*.parquet"))

    def execute(self, context: LCMSContext) -> LCMSContext:
        """Plot every cached sample, then the two cohort overlays; record how many samples were plotted."""
        cache_dir = context.cache_dir / "scan_summary"
        out_dir = context.plots_dir / "chromatograms"
        out_dir.mkdir(parents=True, exist_ok=True)

        scan_dfs = {}
        n_processed = 0
        for parquet_path in cache_dir.glob("*.parquet"):
            sample_id = parquet_path.stem
            df = pd.read_parquet(parquet_path)
            scan_dfs[sample_id] = df

            plot_tic(df, sample_id, out_dir / f"{sample_id}_tic.png")
            plot_bpc(df, sample_id, out_dir / f"{sample_id}_bpc.png")
            n_processed += 1

        # Cohort-scale overlay, in addition to per-sample plots
        sample_metadata = context.sample_metadata.set_index("sample_id")[
            "sample_role"].to_dict() if context.sample_metadata is not None else None
        plot_tic_overlay(scan_dfs, out_dir / "overlay_tic.png", color_by_group=sample_metadata)
        plot_bpc_overlay(scan_dfs, out_dir / "overlay_bpc.png", color_by_group=sample_metadata)

        context.qc_metrics[f"{self._name}"] = {"n_samples_plotted": n_processed}
        context.log_step(
            self._name,
            parameters={},
            metrics={"n_samples_plotted": n_processed},
            input_files=[str(p) for p in cache_dir.glob("*.parquet")],
            output_files=[str(p) for p in out_dir.glob("*.png")],
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The stage left its metrics on the context."""
        return self._name in context.qc_metrics