#!/bin/bash

# === Config ===
H5FILE="/home/robros/labelmaker_test/episode_52.hdf5"

# === 1. Run label.py ===
echo "[1] Running label.py..."
python3 label.py --file "$H5FILE"

if [ $? -ne 0 ]; then
    echo "Error: label.py failed. Aborting."
    exit 1
fi
