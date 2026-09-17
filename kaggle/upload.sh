#!/usr/bin/env bash
# Empaqueta real/ + fake/ y sube el dataset a Kaggle.
# Requiere: haber corrido training/01_prepare_real_data.py y
# training/02_train_generator_and_make_fakes.py primero, y tener
# ~/.kaggle/kaggle.json configurado (ya lo tienes).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="$ROOT/kaggle"

rm -rf "$KAGGLE_DIR/data"
mkdir -p "$KAGGLE_DIR/data/real" "$KAGGLE_DIR/data/fake"

cp "$ROOT"/data/real/*.png "$ROOT"/data/real/*.jpg "$KAGGLE_DIR/data/real/" 2>/dev/null || true
cp "$ROOT"/data/fake/*.png "$KAGGLE_DIR/data/fake/" 2>/dev/null || true
cp "$KAGGLE_DIR/README.md" "$KAGGLE_DIR/data/README.md"

N_REAL=$(ls "$KAGGLE_DIR/data/real" | wc -l | tr -d ' ')
N_FAKE=$(ls "$KAGGLE_DIR/data/fake" | wc -l | tr -d ' ')
echo "Empaquetando $N_REAL reales + $N_FAKE falsas ..."

if [ "$N_REAL" -eq 0 ] || [ "$N_FAKE" -eq 0 ]; then
  echo "Faltan imagenes reales o falsas. Corre primero los scripts 01 y 02 de training/."
  exit 1
fi

# primera vez: kaggle datasets create
# despues de la primera vez: kaggle datasets version -m "mensaje"
if kaggle datasets list -m --user jmmana 2>/dev/null | grep -q "vet-radiographs-real-vs-ai-generated"; then
  kaggle datasets version -p "$KAGGLE_DIR" -m "actualizacion automatica" -d
else
  kaggle datasets create -p "$KAGGLE_DIR" -d
fi

echo "Listo. Revisa https://www.kaggle.com/datasets/jmmana/vet-radiographs-real-vs-ai-generated"
