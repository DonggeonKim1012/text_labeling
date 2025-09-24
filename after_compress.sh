# #!/bin/bash

# # Directory containing your HDF5 files
# TARGET_DIR="/mnt/ddrive/Downloads/with_mask_teleop/with_mask_teleop/vertical"

# # Loop through all *_compressed.h5 files
# for file in "$TARGET_DIR"/*_compressed.h5; do
#     # Skip if no match
#     [ -e "$file" ] || continue

#     echo "Repacking: $file"

#     # Create output file by removing _compressed from the name
#     original_file="${file/_compressed.h5/.h5}"

#     # Use a temporary file for repack output
#     tmpfile="${file%.h5}_repacked.h5"

#     # Repack
#     h5repack "$file" "$tmpfile"

#     # Move repacked file to final output name
#     mv "$tmpfile" "$original_file"

#     # Optionally, remove the original compressed file
#     rm "$file"
# done

# #!/bin/bash

# Directory containing your HDF5 files
TARGET_DIR="/mnt/ddrive/Downloads/with_mask_teleop/with_mask_teleop/vertical"

# Find and remove files ending with _compressed.h5
find "$TARGET_DIR" -type f -name "*.hdf5" -exec rm -v {} \;
