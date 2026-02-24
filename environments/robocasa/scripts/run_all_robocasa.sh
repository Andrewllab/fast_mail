#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

IN_FOLDER_TOP="/mnt/LargeSandwich/kMPD/RoboCasa/v0.1/single_stage"
OUT_FOLDER_TOP="/mnt/LargeSandwich/PAKT/demonstrations/robocasa/single_stage"
IN_FILE_SUFFIX="demo_gentex_im128_randcams.hdf5"
OUT_FILE_SUFFIX="demo_gentex_im128_randcams_preprocessed_1440x1080.hdf5"
CAMERA_HEIGHT=1080
CAMERA_WIDTH=1440

INPUT_FILE="in_files.txt"

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue

  # If your input lines are full paths like:
  # /mnt/.../single_stage/<task>/<date>/demo_gentex_im128_randcams.hdf5
  # then do structured trimming:

  rel="$line"
  rel="${rel#"$IN_FOLDER_TOP"}"        # remove prefix only (safer than //)
  rel="${rel#"/"}"                    # normalize: drop leading slash after prefix removal
  rel="${rel%"$IN_FILE_SUFFIX"}"      # remove trailing suffix only
  rel="${rel%/*/}/"                   # remove date segment between last two slashes

  out_path="$OUT_FOLDER_TOP/$rel$OUT_FILE_SUFFIX"

  mkdir -p "$(dirname "$out_path")"

  echo "IN : $line"
  echo "OUT: $out_path"

  OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python sync_data_states_to_obs.py \
      --dataset "$line" \
      --output_name "$out_path" \
      --camera_height "$CAMERA_HEIGHT" \
      --camera_width "$CAMERA_WIDTH"

done < "$INPUT_FILE"
