---
title: HiveSpace Occupancy Forecast API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# HiveSpace Desk Occupancy Forecast API

FastAPI service that serves daily desk-occupancy forecasts (0–100 desks) from a trained XGBoost model.

## Endpoints

- `GET /` — Service status and supported date range
- `GET /health` — Health check (`{"status": "ok", "model_loaded": true}`)
- `GET /predict?date=YYYY-MM-DD` — Predicted desk occupancy for the requested date
- `GET /docs` — Interactive Swagger UI

## Example

```bash
curl "https://<HF_USERNAME>-hivespace-occupancy-api.hf.space/predict?date=2026-11-20"
```

> **Note:** Trained on synthetic data for demonstration. Predictions more than 1 day past `2026-07-02` are recursive forecasts and include a warning in the JSON response.
