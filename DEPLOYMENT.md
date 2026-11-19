# Deployment Tutorial: HiveSpace Occupancy API on Render (Free Docker Tier)

> [!NOTE]
> **Why Render instead of Hugging Face Spaces?**
> Hugging Face Spaces currently restricts free-tier accounts to **Static** Spaces only (Docker and Gradio SDKs require a paid plan/billing). A Static Space cannot run a Python/FastAPI server. **Render's free Web Service tier** natively builds and runs our exact `Dockerfile` directly from GitHub—with a public HTTPS `/predict` endpoint that works with `curl` and requires **no credit card**.

---

## Step 1: Push Your Latest Code to GitHub

Make sure your latest `Dockerfile`, `app/`, `model/`, `requirements.txt`, and `data/synthetic_occupancy.csv` are pushed to your GitHub repository (`SyedBadiuzzaman/ml_pipeline`):

```powershell
git status
git push origin main
```

---

## Step 2: Create a Free Render Account

1. Go to [https://render.com](https://render.com) and click **Get Started for Free**.
2. Sign in with your **GitHub** account so Render can read your repository directly.

---

## Step 3: Create a New Docker Web Service

1. In the Render Dashboard, click **New +** (top right) → **Web Service**.
2. Connect and select your repository: **`SyedBadiuzzaman/ml_pipeline`**.
3. Fill in the service settings:
   - **Name:** `hivespace-occupancy-api` *(this becomes your URL: `https://hivespace-occupancy-api.onrender.com`)*
   - **Region:** Choose the region closest to you (e.g., Singapore or Frankfurt).
   - **Branch:** `main`
   - **Language / Runtime:** **Docker** *(Render automatically detects your root `Dockerfile`)*
   - **Instance Type:** **Free** (`512 MB RAM · 0.1 CPU · $0/month`)
4. You do **not** need to add any environment variables—our `Dockerfile` automatically listens on Render's `$PORT` (defaulting to `7860` locally).
5. Click **Deploy Web Service**.

---

## Step 4: Watch the Build Logs

1. Render will clone your GitHub repo, run `docker build` using `uv` (typically **1–3 minutes**), and start `uvicorn`.
2. When the log says `Application startup complete` and the badge turns green (**Live**), your API is online!

---

## Step 5: Test the Live Endpoint with `curl`

Replace `<YOUR-RENDER-URL>` with the URL shown at the top of your Render dashboard (for example, `https://hivespace-occupancy-api.onrender.com`):

**Windows PowerShell:**
```powershell
curl.exe "https://<YOUR-RENDER-URL>/health"
curl.exe "https://<YOUR-RENDER-URL>/predict?date=2026-11-20"
```

**macOS / Linux (bash):**
```bash
curl "https://<YOUR-RENDER-URL>/health"
curl "https://<YOUR-RENDER-URL>/predict?date=2026-11-20"
```

Expected response:
```json
{
  "date": "2026-11-20",
  "day_of_week": "Friday",
  "predicted_occupancy": 30,
  "total_desks": 100,
  "forecast_type": "future",
  "horizon_days": 141,
  "warning": "This date is 141 days after the last known data. Only 1-day-ahead accuracy was tested (MAE about 6 desks); error is likely larger this far ahead.",
  "model_version": "xgb-2026-11-19"
}
```

You can also open `https://<YOUR-RENDER-URL>/docs` in your browser to test the interactive Swagger UI.

> [!NOTE]
> On Render's free tier, a service **spins down after ~15 minutes of inactivity**. The first request after spin-down takes ~30–50 seconds while the container wakes up; subsequent requests respond in milliseconds.

---

## Step 6: Update `README.md` with Your Live URL

Once your Render URL is live, send it to me (or replace the placeholder URL in `README.md`), commit, and push to GitHub. Render also auto-redeploys automatically whenever you push new commits to `main`.
