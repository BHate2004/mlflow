#!/bin/bash
set -e

echo ""
echo "=============================================="
echo "  Gurgaon Real Estate ML Pipeline"
echo "=============================================="
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "ERROR: Docker is not running. Please start Docker Desktop and try again."
  exit 1
fi

echo "Step 1/3 — Building images (first time takes ~3 mins)..."
docker compose build --quiet

echo ""
echo "Step 2/3 — Starting all services..."
docker compose up -d

echo ""
echo "Step 3/3 — Waiting for services to be ready..."
echo "(This can take 2-3 minutes on first run)"
echo ""

# Wait for each service
wait_for() {
  local name=$1
  local url=$2
  local max=30
  local i=0
  printf "  Waiting for %-20s" "$name..."
  while ! curl -sf "$url" > /dev/null 2>&1; do
    sleep 5
    i=$((i+1))
    if [ $i -ge $max ]; then
      echo " TIMEOUT (check: docker compose logs $name)"
      return
    fi
    printf "."
  done
  echo " READY ✓"
}

wait_for "MLflow"    "http://localhost:5000/health"
wait_for "FastAPI"   "http://localhost:8000/health"
wait_for "Airflow"   "http://localhost:8080/health"
wait_for "Streamlit" "http://localhost:8501"

echo ""
echo "=============================================="
echo "  All services are running!"
echo "=============================================="
echo ""
echo "  🌐 Streamlit App  →  http://localhost:8501"
echo "  🔌 FastAPI Docs   →  http://localhost:8000/docs"
echo "  📊 MLflow UI      →  http://localhost:5000"
echo "  🌀 Airflow UI     →  http://localhost:8080"
echo "     (login: admin / admin)"
echo ""
echo "  To stop everything: docker compose down"
echo ""
