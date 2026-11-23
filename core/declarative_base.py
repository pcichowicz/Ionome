"""Shared SQLAlchemy declarative base: every ORM model in core.models inherits from `Base`."""
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """Common base class; `Base.metadata` is what `create_all` uses to build the tables."""
    pass
