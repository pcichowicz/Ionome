"""
ORM tables: studies (`Experiment`), their samples (`Sample`) and per-stage run history (`Pipeline`).

The database holds *state* (what exists, what is running, what happened). The scientific results live on disk under
LCMS_RESULTS_DIR/<study_id>/ and are never stored here.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime, UTC
from core.status import STATUS
from core.declarative_base import Base

class Experiment(Base):
    """One study. `study_id` is the slug from the config (`study.id`) used in URLs and folder names."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key = True)
    study_id = Column(String,unique = True , nullable = False)
    created_at = Column(DateTime, default = lambda: datetime.now(UTC))
    config_path = Column(String,nullable = False)       # the registered YAML in YAML_DIR
    status = Column(SQLEnum(STATUS), default=STATUS.PENDING)   # state of the latest run

    samples = relationship("Sample", back_populates="experiment")

class Sample(Base):
    """One raw file belonging to a study. A sample id is unique within its study (see the constraint below)."""
    __tablename__ = "samples"

    id = Column(Integer, primary_key = True, index = True)

    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable = False)
    sample_id = Column(String, index = True, nullable = False)   # equals the raw file name without its extension
    description = Column(String)                                 # filled from the sheet's sample_role column

    status = Column(SQLEnum(STATUS), default=STATUS.PENDING, nullable = False, index=True)

    raw_file_path = Column(String, nullable = False)
    checksum = Column(String, nullable = False)           # SHA-256 of the raw file at registration time

    ingested_at = Column(DateTime, default = lambda: datetime.now(UTC))

    experiment = relationship("Experiment", back_populates="samples")

    @property
    def label(self) -> str:
        """Placeholder: not used anywhere yet (returns a fixed string)."""
        return f"Add in later, might be useful for something"

    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "sample_id",
            name="study_id_sample_id"
        ),
    )

class Pipeline(Base):
    """
    One stage of one run: a run is the set of rows sharing a `run_id`. Rows are written by the stage-event
    recorder in api.services.runs; `error_msg` is set when the stage (or the whole run) failed.
    """
    __tablename__ = "pipeline_runs"
    id = Column(Integer, primary_key = True, index = True)

    run_id = Column(String, index = True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable = False)
    step_name = Column(String, nullable = False)
    status = Column(SQLEnum(STATUS), default=STATUS.PENDING, nullable = False)
    started_at = Column(DateTime, default = lambda: datetime.now(UTC))
    finished_at = Column(DateTime)
    error_msg = Column(String)
