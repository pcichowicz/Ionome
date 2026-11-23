"""Fixed locations inside the repository and inside every study's results folder."""
from pathlib import Path

# Repository root (the folder containing core/, analysis/ and api/).
PROJECT_PATH = Path(__file__).resolve().parents[1]

# The result sub-folders every study gets. The stages' own paths (features/, plots/, ...) are fixed in
# LCMSContext, so this is a constant and not a setting: a different OUTPUT_DIRS could only break them.
RESULT_SUBDIRS = ["features", "logs", "plots", "library", "qc_reports", "cache"]
