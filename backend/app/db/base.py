from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DB_PATH, ensure_directories


class Base(DeclarativeBase):
    pass


ensure_directories()
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_orm() -> None:
    from app.db import models  # noqa: F401

    Base.metadata.create_all(engine)

