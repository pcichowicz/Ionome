"""
Reads the per-sample `.featurejson` files written by feature detection back into Python lists.
"""
from pathlib import Path
import json
from pydantic_core._pydantic_core import ValidationError

from analysis.utils.schemas import Feature
from analysis.pipeline import RecoverableStageError

class JSONFeatureReader:
    """
    Reads back feature lists written by Pyteomics reader.
    """

    def read_features(self, path: Path) -> list[dict]:
        """
        Load and schema-check one feature file.

        Returns:
            A list of dicts with mz, rt, intensity and charge.

        Raises:
            RecoverableStageError: the file is not valid JSON or does not match the Feature schema, so the caller
                can skip that one sample and continue.
        """
        try:
            with open(path) as f:
                raw = json.load(f)

        except json.decoder.JSONDecodeError as exc:
            raise RecoverableStageError(f"{path.name}: corrupt featurejson: {exc}") from exc

        try:
            features = [Feature.model_validate(f).model_dump() for f in raw]
        except ValidationError as exc:
            raise RecoverableStageError(f"{path.name}: featurejson failed schema check: {exc}") from exc

        return features