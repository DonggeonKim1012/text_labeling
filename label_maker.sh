#!/bin/bash

# === Config ===
H5FILE="/mnt/ddrive/Downloads/with_mask_teleop/with_mask_teleop/vertical/episode_88.hdf5"
COMPACTED_FILE="${H5FILE%.hdf5}_compacted.hdf5"

# === 1. Run label.py ===
echo "[1] Running label.py..."
python3 label.py --file "$H5FILE"

if [ $? -ne 0 ]; then
    echo "Error: label.py failed. Aborting."
    exit 1
fi

# === 2. Compact the HDF5 file using h5repack ===
echo "[2] Compacting HDF5 file..."
h5repack "$H5FILE" "$COMPACTED_FILE"

if [ $? -ne 0 ]; then
    echo "Error: h5repack failed. Aborting."
    exit 1
fi

# === 3. Replace the original file ===
echo "[3] Replacing original file with compacted version..."
mv "$COMPACTED_FILE" "$H5FILE"

echo "[✓] Done. File compacted and replaced: $H5FILE"
