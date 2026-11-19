# Deployment Tutorial: HiveSpace Occupancy API on Hugging Face Spaces

This guide walks you step by step from the `Dockerfile` in this repository to a live, public FastAPI endpoint on **Hugging Face Spaces (free CPU tier)**.

All commands below use **Windows PowerShell** from the project root (`ML_Intenship`). Replace `<HF_USERNAME>` with your Hugging Face username.

---

## Step 1: Test the Docker Image Locally First

Always verify the container on your own machine before pushing to Hugging Face. Fixing errors locally takes seconds instead of waiting for remote builds.

```powershell
# 1. Build the Docker image
docker build -t hivespace-api .

# 2. Run the container on port 7860
docker run --rm -p 7860:7860 hivespace-api
```

Open a **second** PowerShell window and test the endpoints with `curl.exe`:

```powershell
curl.exe "http://localhost:7860/health"
curl.exe "http://localhost:7860/predict?date=2026-07-03"
curl.exe "http://localhost:7860/predict?date=2026-11-20"
```

You can also open `http://localhost:7860/docs` in your browser to try the interactive Swagger UI. Press `Ctrl + C` in the first terminal when finished.

---

## Step 2: Create a Free Hugging Face Account

1. Go to [https://huggingface.co/join](https://huggingface.co/join) and sign up.
2. Verify your email address.

---

## Step 3: Create a New Docker Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Fill in the form:
   - **Space name:** `hivespace-occupancy-api`
   - **Select the Space SDK:** choose **Docker**, then select the **Blank** template.
   - **Space hardware:** `CPU basic · 2 vCPU · 16 GB RAM · Free`
   - **Visibility:** **Public** (so your mentor can test the live endpoint with `curl`).
3. Click **Create Space**.

---

## Step 4: Create a Hugging Face Write Access Token

Hugging Face requires a personal access token (not your account password) when pushing with `git`.

1. Click your profile picture (top right) → **Settings** → **Access Tokens**.
2. Click **Create new token**.
3. Choose **Write** permissions, give it a name like `hivespace-deploy`, and click **Create token**.
4. Copy the token somewhere safe for Step 7.

> [!CAUTION]
> **Never** paste your access token into any code file, `README.md`, or commit. Only enter it when `git push` asks for a password.

---

## Step 5: Clone the Empty Space Repository Next to Your Project

Clone the Space into the **parent folder** (next to `ML_Intenship`, not inside it):

```powershell
cd ..
git clone https://huggingface.co/spaces/<HF_USERNAME>/hivespace-occupancy-api
cd ML_Intenship
```

---

## Step 6: Copy Only the Files the API Needs

Hugging Face rejects standard `git push` commits that include binary image files (like `data/backtest_predictions.png`). Copying only the files needed for serving keeps the Space clean, small, and error-free:

```powershell
$space = "..\hivespace-occupancy-api"

Copy-Item Dockerfile, .dockerignore, requirements.txt $space
Copy-Item deploy\space_README.md "$space\README.md"
Copy-Item app, model -Destination $space -Recurse -Force

New-Item -ItemType Directory -Force "$space\pipeline", "$space\data" | Out-Null
Copy-Item pipeline\__init__.py, pipeline\process_split.py "$space\pipeline\"
Copy-Item data\synthetic_occupancy.csv "$space\data\"
```

*(Optional: open `$space\README.md` and replace `<HF_USERNAME>` with your username.)*

---

## Step 7: Commit and Push to Hugging Face Spaces

```powershell
cd ..\hivespace-occupancy-api
git add .
git commit -m "Deploy HiveSpace desk occupancy FastAPI service"
git push
```

When Git prompts for credentials:
- **Username:** your Hugging Face username
- **Password:** paste the **Write** token from Step 4

---

## Step 8: Watch the Build Logs

1. Open your Space page: `https://huggingface.co/spaces/<HF_USERNAME>/hivespace-occupancy-api`
2. Click the **Logs** tab (Build + Container tabs).
3. Wait 2–5 minutes while Hugging Face builds the Docker image and starts `uvicorn` on port `7860`.
4. When the status badge turns green (**Running**), your API is live!

**Common build issues:**
- *File not found during `COPY`:* re-run Step 6 and verify `model/xgb_occupancy.json` exists (`python pipeline/export_model.py`).
- *Push rejected for binary files:* make sure you did **not** copy `data/backtest_predictions.png` into the Space folder.

---

## Step 9: Verify the Live Endpoint with `curl`

Hugging Face exposes your Docker Space at:
`https://<HF_USERNAME>-hivespace-occupancy-api.hf.space`

Test it from PowerShell:

```powershell
curl.exe "https://<HF_USERNAME>-hivespace-occupancy-api.hf.space/health"
curl.exe "https://<HF_USERNAME>-hivespace-occupancy-api.hf.space/predict?date=2026-11-20"
```

Expected response:
```json
{
  "date": "2026-11-20",
  "day_of_week": "Friday",
  "predicted_occupancy": 47,
  "total_desks": 100,
  "forecast_type": "future",
  "horizon_days": 141,
  "warning": "This date is 141 days after the last known data. Only 1-day-ahead accuracy was tested (MAE about 6 desks); error is likely larger this far ahead.",
  "model_version": "xgb-2026-11-19"
}
```

You can also open `https://<HF_USERNAME>-hivespace-occupancy-api.hf.space/docs` in your browser to use the interactive Swagger UI.

> [!NOTE]
> On Hugging Face's free CPU tier, a Space **sleeps after ~48 hours of inactivity**. The first request after sleep takes ~30–60 seconds while the container wakes up; subsequent requests respond in milliseconds.

---

## Step 10: Update Your GitHub README with the Live URL

Once verified, replace `<HF_USERNAME>` in `README.md` with your real Hugging Face username and push the update to GitHub so your mentor can click and `curl` the live URL directly.

## Redeploying After Model Updates

Whenever you retrain or update the data:
```powershell
python pipeline/export_model.py
# Then repeat Steps 6 and 7 to push the updated model/ and data/ to the Space.
```
