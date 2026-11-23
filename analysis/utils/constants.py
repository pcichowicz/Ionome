"""
Constants shared by several stages.
"""
# Suffix of the per-sample feature files written by feature_detection (features/<sample_id>.featurejson)
FEATURE_FILE_SUFFIX = ".featurejson"

# Columns of the aligned feature table that are NOT sample intensities. Anything else is treated as a sample
# column, so a new annotation column must be added here or it would be mistaken for a sample.
NON_SAMPLE_COLS = ("feature_id", "mz", "rt", "blank_level", "blank_flagged",
                   "adduct_group_id", "ion_type", "neutral_mass", "ion_evidence")