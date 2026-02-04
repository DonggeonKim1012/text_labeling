import os
import h5py
import numpy as np
import cv2
import argparse
import subprocess
import tempfile
import shutil

# ======================
# Settings
# ======================
FPS = 60
FRAME_SIZE = (1280, 480)
BATCH_SIZE = 1  # number of HDF5 files per video

# Overlay settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_ORIGIN = (10, 30)


def draw_label(img, text):
    """Overlay episode name on frame."""
    (w, h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x, y = TEXT_ORIGIN
    cv2.rectangle(img, (x - 5, y - h - 5), (x + w + 5, y + baseline + 5), TEXT_BG_COLOR, -1)
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def hdf5_to_mp4_ffmpeg(hdf5_folder, output_folder, batch_size=BATCH_SIZE, fps=FPS):
    """Convert HDF5 head_camera frames to batched MP4 videos using FFmpeg."""
    os.makedirs(output_folder, exist_ok=True)

    hdf5_files = sorted(f for f in os.listdir(hdf5_folder) if f.endswith((".h5", ".hdf5")))
    if not hdf5_files:
        raise RuntimeError("No HDF5 files found in the input folder.")

    total_batches = (len(hdf5_files) + batch_size - 1) // batch_size
    print(f"Total files: {len(hdf5_files)}, total batches: {total_batches}")

    for batch_idx in range(total_batches):
        batch_files = hdf5_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        output_video_path = os.path.join(output_folder, f"combined_{batch_idx:03d}.mp4")

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_counter = 0
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches}, {len(batch_files)} files...")

            for fname in batch_files:
                h5_path = os.path.join(hdf5_folder, fname)
                episode_label = os.path.splitext(fname)[0]

                try:
                    with h5py.File(h5_path, "r") as f:
                        if "observations/images/head_camera" not in f:
                            print(f"Skipping {fname}, dataset not found.")
                            continue

                        ds = f["observations/images/head_camera"]
                        for i in range(len(ds)):
                            frame_bytes = ds[i]
                            buf = (
                                np.frombuffer(frame_bytes, dtype=np.uint8)
                                if isinstance(frame_bytes, bytes)
                                else frame_bytes.flatten()
                            )
                            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            if img is None:
                                print(f"Warning: Could not decode frame {i} in {fname}, skipping.")
                                continue

                            if img.shape[:2] != FRAME_SIZE[::-1]:
                                img = cv2.resize(img, FRAME_SIZE)
                            draw_label(img, episode_label)

                            # Save as PNG
                            frame_file = os.path.join(tmpdir, f"frame_{frame_counter:06d}.png")
                            cv2.imwrite(frame_file, img)
                            frame_counter += 1

                            if frame_counter % 100 == 0:
                                print(f"Saved {frame_counter} frames...")

                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                    continue

            print(f"All frames saved. Total frames: {frame_counter}")
            if frame_counter == 0:
                print(f"No frames found in batch {batch_idx+1}, skipping video creation.")
                continue

            # Call FFmpeg to create MP4
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # overwrite output
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]
            print(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True)
            print(f"Batch {batch_idx+1} video saved to {output_video_path}")


# ======================
# Main
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 head_camera frames to batched MP4 videos using FFmpeg")
    parser.add_argument("--input_folder", required=True, help="Folder containing HDF5 files")
    parser.add_argument("--output_folder", required=True, help="Folder to save MP4 videos")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Number of files per video")
    parser.add_argument("--fps", type=int, default=FPS, help="Frames per second for output videos")
    args = parser.parse_args()

    hdf5_to_mp4_ffmpeg(args.input_folder, args.output_folder, args.batch_size, args.fps)

