#!/bin/bash
# ─────────────────────────────────────────────────────────
#  PetFinder EDA — Run Pipeline & Launch Dashboard
# ─────────────────────────────────────────────────────────
set -e

PORT=${1:-8081}

echo "══════════════════════════════════════════════"
echo "  P4AI-DS EDA Pipeline"
echo "══════════════════════════════════════════════"

echo ""
echo "[1/5] Running text EDA export..."
uv run python scripts/eda_text.py

echo ""
echo "[2/5] Running tabular EDA export..."
uv run python scripts/eda_salary.py

echo ""
echo "[3/5] Running EDA data export..."
uv run python scripts/eda_image.py

echo ""
echo "[4/5] Building image gallery..."
uv run python scripts/gallery_export.py

echo ""
echo "[5/5] Starting web server on http://localhost:${PORT}"
echo "      Press Ctrl+C to stop."
echo ""
cd ui && python3 -m http.server "$PORT"
