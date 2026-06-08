#!/usr/bin/env bash
# Lance Milimo Video avec les modèles QUANTIFIÉS 4-bit (tient sur la RTX 3090) :
#   - transformer LTX-2 19B  -> bitsandbytes 4-bit (build depuis le bf16 local)
#   - gemma-3 12B            -> bitsandbytes 4-bit
# Démarre le backend (FastAPI, :8010) + le frontend (Vite, :5173).
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/milimov"
PY="$VENV/bin/python"

# --- env quantification (lus par le ModelLedger / load_gemma au chargement) ---
export MILIMO_TRANSFORMER_BNB=1     # transformer en 4-bit bitsandbytes
export MILIMO_GEMMA_4BIT=1          # text encoder gemma en 4-bit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr
export PATH="$VENV/bin:/usr/bin:$PATH"

[ -x "$PY" ] || { echo "venv introuvable: $PY"; exit 1; }

LAN_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^192\.168\.|^10\.' | head -1)"
PIDS=()
cleanup() { echo; echo "Arrêt..."; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; exit 0; }
trap cleanup INT TERM

# --- Backend (FastAPI, port 8010, déjà bindé 0.0.0.0 dans server.py) ---
echo "==> Backend (port 8010)..."
( cd "$DIR/backend" && exec "$PY" server.py ) &
PIDS+=($!)

# --- Frontend (Vite, port 5173) ---
if [ ! -d "$DIR/web-app/node_modules" ]; then
  echo "==> Première fois : npm install (quelques minutes)..."
  ( cd "$DIR/web-app" && npm install )
fi
echo "==> Frontend (port 5173)..."
( cd "$DIR/web-app" && exec npm run dev -- --host 0.0.0.0 ) &
PIDS+=($!)

sleep 2
cat <<EOF

============================================================
  Milimo Video — modèles 4-bit chargés à la 1ère génération
============================================================
  Interface :  http://localhost:5173        (sur cette machine)
  Backend   :  http://localhost:8010

  Accès DISTANT (ton autre PC) — tunnel SSH (le frontend
  appelle localhost:8010, donc tunnelise les 2 ports) :

    ssh -L 5173:localhost:5173 -L 8010:localhost:8010 pc@${LAN_IP:-<IP_SERVEUR>}

  puis ouvre  http://localhost:5173  dans ton navigateur.

  (Ctrl+C ici arrête backend + frontend.)
  Note: la 1ère génération charge le transformer (~3-4 min,
  build bnb depuis le bf16 local) ; les suivantes réutilisent.
============================================================
EOF

wait
