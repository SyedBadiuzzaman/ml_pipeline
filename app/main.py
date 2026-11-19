"""FastAPI web service for HiveSpace daily desk occupancy forecasts.

Endpoints:
- GET /                   Welcome message with links to docs and an example call.
- GET /health             Health check confirming the model is loaded.
- GET /predict?date=...   Predicted desk occupancy for a YYYY-MM-DD date.
- GET /docs               Interactive Swagger UI (built into FastAPI).

Run locally from the project root:
    uvicorn app.main:app --host 0.0.0.0 --port 7860
"""

from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.forecast import initialize_forecast_state


class PredictionResponse(BaseModel):
    """JSON response returned by GET /predict."""

    date: date
    day_of_week: str
    predicted_occupancy: int
    total_desks: int
    forecast_type: str
    horizon_days: int
    warning: str | None = None
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved XGBoost model and precompute the forecast table once at startup."""
    app.state.forecast = initialize_forecast_state()
    yield


app = FastAPI(
    title="HiveSpace Desk Occupancy Forecast API",
    description=(
        "Predicts daily HiveSpace desk occupancy (0-100 desks) using a trained "
        "XGBoost model. Trained on synthetic data for demonstration purposes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    """Landing endpoint with quick links for managers and developers."""
    state = app.state.forecast
    return {
        "service": "HiveSpace Desk Occupancy Forecast API",
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health",
        "example_request": "/predict?date=2026-07-03",
        "supported_date_range": {
            "min_date": str(state["min_date"]),
            "max_date": str(state["max_date"]),
            "last_known_data_date": str(state["last_data_date"]),
        },
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint used by Hugging Face Spaces and curl checks."""
    state = app.state.forecast
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "model_version": state["model_info"]["model_version"],
        "cached_dates": len(state["forecast_table"]),
    }


@app.get("/predict", response_model=PredictionResponse)
def predict(
    target_date: date = Query(
        ...,
        alias="date",
        description="Target date in YYYY-MM-DD format (e.g., 2026-11-20).",
    ),
) -> PredictionResponse:
    """Return the predicted desk occupancy for the requested date."""
    state = app.state.forecast
    entry = state["forecast_table"].get(target_date)

    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Date {target_date} is outside the supported range "
                f"({state['min_date']} to {state['max_date']})."
            ),
        )

    return PredictionResponse(
        **entry,
        model_version=state["model_info"]["model_version"],
    )
