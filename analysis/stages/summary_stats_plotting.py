"""
Summary statistics plots (stage 10 of cohort_with_qc only): data-quality overview of the whole cohort.

Reads the annotated (or, if absent, aligned) feature table and writes to `<results>/plots/summary_stats/`:
  * `rsd_distribution.png`: how reproducible each feature is across the QC replicates (needs >= 2 QC samples);
  * `pca_scores.png`: PCA of all samples; QC samples should cluster tightly (needs >= 3 samples and >= 2 features).
A plot that cannot be made is skipped with a warning. Samples are grouped by `sample_role`; QC samples are those whose
role is exactly "QC".
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from analysis.context import LCMSContext
from analysis.pipeline import FatalStageError
from analysis.utils.constants import NON_SAMPLE_COLS
from analysis.utils.plotting import plot_rsd_distribution, plot_pca_scores, prepare_pca_matrix, compute_feature_rsd,compute_pca_scores

class SummaryStatsPlottingStage:
    """Create the QC RSD histogram and the PCA scores plot (see the module docstring)."""
    _name = "summary_stats_plotting"
    # requires: stages whose results this one reads (must be registered earlier; checked before anything runs).
    # runs_after: stages it only reads if present; if they are registered they must come earlier.
    requires: tuple[str, ...] = ('feature_alignment',)

    def __init__(self, rsd_threshold: float = 20.0, min_presence_frac: float = 0.5):
        """
        Args:
            rsd_threshold: percent RSD drawn as the cut-off line (and used to report the share of features above it).
            min_presence_frac: a feature enters the PCA only if detected in at least this fraction of samples (0 to 1).
        """
        self.rsd_threshold = rsd_threshold
        self.min_presence_frac = min_presence_frac

    def _feature_table_path(self, context: LCMSContext) -> Path:
        """The annotated table when adduct_annotation produced one, otherwise the aligned table."""
        annotated = context.featurejson_dir / f"{context.study_id}_annotated_features.parquet"
        aligned = context.featurejson_dir / f"{context.study_id}_aligned_features.parquet"
        return annotated if annotated.exists() else aligned

    def validate_input(self, context: LCMSContext) -> bool:
        """Fatal if no feature table exists or the sample sheet is not loaded."""
        if not self._feature_table_path(context).exists():
            raise FatalStageError(f"summary_stats_plotting: no feature table found at {self._feature_table_path(context)}")
        if context.sample_metadata is None:
            raise FatalStageError("summary_stats_plotting: sample_metadata not loaded on context")
        return True

    def execute(self, context: LCMSContext) -> LCMSContext:
        """Make the RSD and PCA plots and record their numbers (share above the RSD threshold, explained variance)."""
        table = pd.read_parquet(self._feature_table_path(context))
        sample_cols = [c for c in table.columns if c not in NON_SAMPLE_COLS]

        role_map = context.sample_metadata.set_index("sample_id")["sample_role"].to_dict()
        qc_sample_cols = [c for c in sample_cols if role_map.get(c) == "QC"]

        out_dir = context.plots_dir / "summary_stats"
        out_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []

        # --- RSD ---
        if len(qc_sample_cols) < 2:
            warnings.append(f"summary_stats_plotting: only {len(qc_sample_cols)} QC sample(s) found, RSD skipped")
            rsd_result = {"n_features_scored": 0, "pct_above_threshold": None}
        else:
            rsd_values = compute_feature_rsd(table, qc_sample_cols)
            plot_rsd_distribution(rsd_values, self.rsd_threshold, out_dir / "rsd_distribution.png")
            valid = rsd_values.dropna()
            rsd_result = {
                "n_features_scored": int(len(valid)),
                "pct_above_threshold": float((valid.to_numpy() > self.rsd_threshold).mean() * 100) if len(valid) else None,
                "median_rsd": float(valid.median()) if len(valid) else None,
            }

        # --- PCA ---
        matrix = prepare_pca_matrix(table, sample_cols, self.min_presence_frac)
        if matrix.shape[1] < 2 or matrix.shape[0] < 3:
            warnings.append("summary_stats_plotting: not enough features/samples for PCA, skipped")
            pca_result = {"n_features_used": matrix.shape[1], "explained_variance": None}
        else:
            scores_df, explained_variance = compute_pca_scores(matrix)
            plot_pca_scores(scores_df, role_map, explained_variance, out_dir / "pca_scores.png")
            pca_result = {
                "n_features_used": int(matrix.shape[1]),
                "explained_variance_pc1": float(explained_variance[0]),
                "explained_variance_pc2": float(explained_variance[1]),
            }

        context.qc_metrics[self._name] = {"rsd": rsd_result, "pca": pca_result}
        context.log_step(
            self._name,
            parameters={"rsd_threshold": self.rsd_threshold,
                        "min_presence_frac": self.min_presence_frac},
            metrics=context.qc_metrics[self._name],
            warnings=warnings,
            input_files=[str(self._feature_table_path(context))],
            output_files=[str(p) for p in out_dir.glob("*.png")],
        )
        return context

    def validate_output(self, context: LCMSContext) -> bool:
        """The stage left its metrics on the context."""
        return self._name in context.qc_metrics