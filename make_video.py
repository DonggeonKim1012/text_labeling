import h5py
import numpy as np
import cv2

def decompress_and_reshape_to_rgb_video(
    hdf5_file_path,
    output_video_path='output_video.mp4',
    fps=30,
    original_height=480,
    original_width=640,
):
    """
    Reads compressed image data from an HDF5 file (compressed by cv2.imencode),
    decompresses, reshapes to (3, H, W) RGB, and creates a video.

    Args:
        hdf5_file_path (str): Path to your HDF5 file.
        output_video_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.
        original_height (int): Original height of each image frame (e.g., 480).
        original_width (int): Original width of each image frame (e.g., 640).
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as f:
            dataset_name = 'observations/images/head_camera'
            if dataset_name in f:
                raw_data = f[dataset_name][:]

                t = raw_data.shape[0]

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (original_width, original_height), isColor=True)

                if not out.isOpened():
                    print(f"Error: Could not open video writer for {output_video_path}")
                    return

                print(f"Processing {t} frames...")

                for i in range(t):
                    single_frame_compressed_bytes = raw_data[i]

                    # Decompress using cv2.imdecode
                    # cv2.IMREAD_GRAYSCALE or cv2.IMREAD_COLOR depending on original image
                    # Assuming your original images before imencode were grayscale:
                    decoded_frame = cv2.imdecode(single_frame_compressed_bytes, cv2.IMREAD_COLOR)

                    if decoded_frame is None:
                        print(f"Warning: Could not decode frame {i}. Skipping.")
                        continue

                    # Reshape (though imdecode should directly give (H,W) or (H,W,3))
                    # Check if it's grayscale and needs conversion to 3-channel
                    if len(decoded_frame.shape) == 2: # It's a grayscale image
                        # Convert grayscale to 3-channel (RGB/BGR) for video writer
                        color_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_GRAY2BGR)
                    elif len(decoded_frame.shape) == 3 and decoded_frame.shape[2] == 3: # It's already 3-channel
                        color_frame = decoded_frame[:,:640]
                    else:
                        print(f"Warning: Frame {i} has unexpected number of channels: {decoded_frame.shape}. Skipping.")
                        continue
                    
                    # Ensure the frame has the correct dimensions
                    if color_frame.shape[0] != original_height or color_frame.shape[1] != original_width:
                        print(f"Warning: Frame {i} has incorrect dimensions {color_frame.shape}. Expected ({original_height}, {original_width}, 3). Resizing.")
                        color_frame = cv2.resize(color_frame, (original_width, original_height))

                    color_frame = color_frame.astype(np.uint8)

                    out.write(color_frame)

                    if (i + 1) % 100 == 0:
                        print(f"Processed frame {i + 1}/{t}")

                out.release()
                print(f"Video saved to {output_video_path}")

            else:
                print(f"Dataset '{dataset_name}' not found in {hdf5_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    dummy_hdf5_path = '/mnt/ddrive/Downloads/with_mask_teleop/with_mask_teleop/episode_13.hdf5'
    num_frames_to_create = 50
    frame_height = 480
    frame_width = 640

    output_video_file = 'image_video.mp4'
    decompress_and_reshape_to_rgb_video(dummy_hdf5_path, output_video_file, fps=100,
                                        original_height=frame_height, original_width=frame_width)
