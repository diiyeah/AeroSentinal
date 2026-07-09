"""
AeroSentinal — FastAPI Backend Entry Point
===========================================
Main application with CORS, health check, and subsystem routers.

Start with:
  uvicorn backend.app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import engine, hydraulics, landing_gear, apu, ecs
from backend.app.routers import fusion, simulator, nlp

app = FastAPI(
    title="AeroSentinal",
    description=(
        "Holistic Aircraft Predictive Maintenance API. "
        "Provides per-subsystem predictions, fusion scoring, "
        "what-if simulation, ACARS alert compilation, and AOG risk assessment.\n\n"
        "**Honesty Policy:** All endpoints report `is_synthetic_data: true` "
        "when using physics-informed simulation rather than trained ML models. "
        "This flag will flip to `false` when ONNX models from real training runs "
        "are deployed."
    ),
    version="0.2.0-prototype",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    from backend.app.services.inference import (
        EngineService,
        HydraulicsService,
        LandingGearService,
        APUService,
    )

    return {
        "status": "healthy",
        "service": "AeroSentinal Backend",
        "version": "0.2.0-prototype",
        "subsystems": {
            "engine": "model_loaded" if EngineService.MODEL_LOADED else "simulation",
            "hydraulics": "model_loaded" if HydraulicsService.MODEL_LOADED else "simulation",
            "landing_gear": "model_loaded" if LandingGearService.MODEL_LOADED else "simulation",
            "apu": "model_loaded" if APUService.MODEL_LOADED else "simulation",
            "ecs": "physics_model",  # ECS is always a physics model by design
        },
        "endpoints": {
            "docs": "/docs",
            "predict_engine": "/predict/engine/quick",
            "predict_hydraulics": "/predict/hydraulics/quick",
            "predict_landing_gear": "/predict/landing-gear",
            "predict_apu": "/predict/apu",
            "predict_ecs": "/predict/ecs",
            "fusion_health": "/fusion/health",
            "acars": "/fusion/acars",
            "simulator": "/simulate/what-if",
        },
    }


# ============================================================
# Register Routers
# ============================================================

# Individual subsystem prediction endpoints
app.include_router(engine.router, prefix="/predict", tags=["Engine"])
app.include_router(hydraulics.router, prefix="/predict", tags=["Hydraulics"])
app.include_router(landing_gear.router, prefix="/predict", tags=["Landing Gear"])
app.include_router(apu.router, prefix="/predict", tags=["APU"])
app.include_router(ecs.router, prefix="/predict", tags=["ECS"])

# Fusion orchestration + ACARS
app.include_router(fusion.router, prefix="/fusion", tags=["Fusion"])

# What-if simulator
app.include_router(simulator.router, prefix="/simulate", tags=["Simulator"])

# NLP repair recommendations (Phase 1 stand-in)
app.include_router(nlp.router, prefix="/nlp", tags=["NLP"])
