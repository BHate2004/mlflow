# How to Run — Gurgaon Real Estate ML Pipeline

## Requirements (install these first)
- **Docker Desktop**: https://www.docker.com/products/docker-desktop/
  - Make sure Docker Desktop is **open and running** before doing anything else

---

## Run the project (3 steps only)

### Step 1 — Open a terminal in the project folder

On **Mac/Linux**: right-click the folder → "Open Terminal"  
On **Windows**: right-click the folder → "Open in Terminal" or open PowerShell and run:
```
cd path\to\real-estate-ml-pipeline-fixed
```

### Step 2 — Start everything with one command

**Mac/Linux:**
```bash
./start.sh
```

**Windows (PowerShell):**
```powershell
docker compose build
docker compose up -d
```

### Step 3 — Open your browser

Wait ~2-3 minutes, then open:

| What | URL | Login |
|------|-----|-------|
| 🌐 **Streamlit App** (main UI) | http://localhost:8501 | — |
| 🔌 **FastAPI** (API + auto docs) | http://localhost:8000/docs | — |
| 📊 **MLflow** (experiment tracking) | http://localhost:5000 | — |
| 🌀 **Airflow** (pipeline scheduler) | http://localhost:8080 | admin / admin |

---

## Stop everything
```bash
./stop.sh
```
Or manually:
```bash
docker compose down
```

---

## Common Fixes

### "Port already in use" error
Another app is using that port. Stop the conflicting app, or run:
```bash
docker compose down
docker compose up -d
```

### Containers keep restarting
Check logs:
```bash
docker compose logs airflow-webserver
docker compose logs api
```

### Start fresh (wipe all data)
```bash
docker compose down -v
docker compose up -d
```

### Check what's running
```bash
docker compose ps
```

---

## What each service does

- **Streamlit** — the web app where you enter property details and get a price estimate
- **FastAPI** — the backend API that runs the ML model
- **MLflow** — tracks model training experiments and stores model versions
- **Airflow** — runs scheduled pipelines (daily monitoring + weekly retraining)
- **MariaDB** — database that stores property data
- **Redis** — caching layer for faster predictions
