import h5py
import cv2
import numpy as np

def make_mask_video_from_h5(h5_path, dataset_path='prompts/masks/head_camera', output_path='masks_video.mp4', fps=30):
    with h5py.File(h5_path, 'r') as f:
        masks = f[dataset_path][:]
        T, H, W = masks.shape
        print(f"Loaded {T} masks of size {H}x{W}")

    # Create a color video: black background, green where mask == 255
    height, width = H, W
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(T):
        mask = masks[i]

        # Convert binary mask to 3-channel RGB (green where mask == 255)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[mask == 255] = [0, 255, 0]  # green mask

        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    hdf5_path = '/mnt/ddrive/Downloads/inbox_episode_1.hdf5'
    output_video_path = 'masks_video.mp4'
    make_mask_video_from_h5(hdf5_path)