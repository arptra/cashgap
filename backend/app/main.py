from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.datasets import router as datasets_router
from app.api.jobs import router as jobs_router
from app.api.experiments import router as experiments_router
from app.api.sources import router as sources_router
from app.api.synthetic import router as synthetic_router
from app.api.training import router as training_router
from app.api.models import router as models_router
from app.api.benchmarks import router as benchmarks_router
from app.config import ensure_directories
from app.db.base import init_orm
from app.services.registry import seed_registry
from app.storage.database import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_directories()
    init_db()
    init_orm()
    seed_registry()
    yield


app = FastAPI(
    title="CashGap Lab API",
    version="1.0.0",
    description="Local synthetic-data and cash-gap risk experimentation stand.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(datasets_router, prefix="/api")
app.include_router(sources_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")
app.include_router(synthetic_router, prefix="/api")
app.include_router(experiments_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(benchmarks_router, prefix="/api")


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "cashgap-lab"}
