"""Run/sample lifecycle states shared by the database models, the pipeline recorder and the API."""
from enum import Enum

class STATUS(Enum):
    """
    State of an experiment, a sample, or one pipeline stage.

    PENDING -> RUNNING -> SUCCESS | FAILED. DONE is not used anywhere in the code; SUCCESS is the final good state.
    """
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    DONE = "DONE"
    FAILED = "FAILED"
