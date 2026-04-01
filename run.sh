#!/bin/bash
# ─────────────────────────────────────────────────────────
#  PetFinder EDA — Run Pipeline & Launch Dashboard
# ─────────────────────────────────────────────────────────
set -e

PORT=${1:-8080}

echo "══════════════════════════════════════════════"
echo "  P4AI-DS EDA Pipeline"
echo "══════════════════════════════════════════════"

# Step 1: Run EDA data export
echo ""
echo "[1/3] Running EDA data export..."
uv run python scripts/eda_image.py

# Step 2: Run gallery export
echo ""
echo "[2/3] Building image gallery..."
uv run python scripts/gallery_export.py

# Step 3: Start web server
echo ""
echo "[3/3] Starting web server on http://localhost:${PORT}"
echo "      Press Ctrl+C to stop."
echo ""
cd ui && python3 -m http.server "$PORT"
