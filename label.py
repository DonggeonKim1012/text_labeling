import h5py
import cv2
import zlib
import numpy as np
import time
import subprocess
import torch
from transformers import AutoTokenizer
import argparse

def decompress_mask(mask_bytes, shape=(480, 640), dtype=np.uint8):
    decompressed = zlib.decompress(mask_bytes)
    return np.frombuffer(decompressed, dtype=dtype).reshape(shape)

def decompress_mask2(mask, shape=(480, 640), dtype=np.uint8):
    if isinstance(mask, np.ndarray):
        return mask
    elif isinstance(mask, (bytes, bytearray)):
        decompressed = zlib.decompress(mask)
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
    else:
        raise ValueError("Unsupported mask type.")

def detect_mask_changes(
    masks,
    h5file=None,
    dataset_name='/prompts/masks/head_camera',
    threshold_appear=1000,
    threshold_disappear=1000,
    overlap_threshold=100
):
    print("Decompressing masks and detecting changes...")
    T = len(masks)
    decompressed_masks = np.empty((T, 480, 640), dtype=np.uint8)
    change_frames = []
    diffs_total, diffs_appear, diffs_disappear = [], [], []

    # Use correct decompressor
    decompress = decompress_mask2 if masks[0].ndim == 2 else decompress_mask

    for i in range(T):
        mask = decompress(masks[i])
        decompressed_masks[i] = mask

        if i > 0:
            prev_mask = decompressed_masks[i - 1]

            appear_mask = (prev_mask == 0) & (mask == 255)
            disappear_mask = (prev_mask == 255) & (mask == 0)
            overlap_mask = (prev_mask == 255) & (mask == 255)

            diff_appear = np.count_nonzero(appear_mask)
            diff_disappear = np.count_nonzero(disappear_mask)
            overlap_count = np.count_nonzero(overlap_mask)
            total_diff = diff_appear + diff_disappear

            if (total_diff - overlap_count > overlap_threshold) or ((diff_disappear == 0 and diff_appear > 200) ^ (diff_appear == 0 and diff_disappear > 60)):
                change_frames.append(i)
                diffs_total.append(total_diff)
                diffs_appear.append(diff_appear)
                diffs_disappear.append(diff_disappear)

    if h5file is not None:
        if dataset_name in h5file:
            del h5file[dataset_name]
        h5file.create_dataset(dataset_name, data=decompressed_masks, dtype='uint8', compression='gzip', compression_opts=9)
        print(f"Decompressed masks saved (with compression) to {dataset_name}")


    print("Changes at change frames:")
    for idx, frame in enumerate(change_frames):
        print(f"Frame {frame}: appear = {diffs_appear[idx]}, disappear = {diffs_disappear[idx]}")

    return change_frames

def show_image_with_mask(image_bytes, mask_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)[:, :640]
    mask = decompress_mask2(mask_bytes)
    color_mask = np.zeros_like(image)
    color_mask[mask == 255] = [0, 255, 0]
    overlayed = cv2.addWeighted(image, 0.8, color_mask, 0.2, 0)
    cv2.imshow('Changed Frame with Mask', overlayed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_three_frames(images, masks, idx):
    T = len(images)
    idxs = [max(idx - 1, 0), idx, min(idx + 1, T - 1)]
    decompress = decompress_mask2 if masks[0].ndim == 2 else decompress_mask

    frames = []
    for i in idxs:
        image_bytes = images[i]
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)[:, :640]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = decompress(masks[i])
        overlay = image.copy()
        overlay[mask == 255] = [0, 255, 0]
        frames.append(overlay)

    combined = np.hstack(frames)
    window_name = f"Frames {idxs[0]}, {idxs[1]}, {idxs[2]}"
    cv2.imshow(window_name, combined)
    cv2.waitKey(1)

    time.sleep(0.1)
    try:
        subprocess.run(["xdotool", "search", "--name", window_name, "windowactivate"], check=True)
    except Exception as e:
        print(f"[경고] 창 포그라운드 이동 실패: {e}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to HDF5 file')
    args = parser.parse_args()
    file_path = args.file
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')

    with h5py.File(file_path, 'a') as f:
        if 'prompts/masks/head_camera' in f:
            print("Using decompressed masks from /prompts/masks/head_camera")
            masks = f['prompts/masks/head_camera'][:]
        else:
            print("Using compressed masks from /observations/masks/head_camera")
            masks = f['observations/masks/head_camera'][:]

        images = f['observations/images/head_camera'][:]
        T = images.shape[0]

        change_frames = detect_mask_changes(masks, h5file=f, threshold_appear=400, threshold_disappear=200, overlap_threshold=800)

        current_idx = 0
        current_prompt = ""
        empty_tokenized = tokenizer(
            current_prompt,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=32
        )
        max_seq_length = empty_tokenized["input_ids"].shape[1]

        input_ids_buffer = np.full((T, max_seq_length), empty_tokenized["input_ids"].numpy(), dtype=np.int64)
        token_type_ids_buffer = np.full((T, max_seq_length), empty_tokenized["token_type_ids"].numpy(), dtype=np.int64)
        attention_mask_buffer = np.full((T, max_seq_length), empty_tokenized["attention_mask"].numpy(), dtype=np.int64)

        prev_prompt_saved = True

        while current_idx < len(change_frames):
            idx = change_frames[current_idx]
            show_three_frames(images, masks, idx)

            user_input = input(f"Enter prompt for frame {idx}: ").strip()

            if user_input.lower() == 'back':
                if current_idx > 0:
                    current_idx -= 1
                else:
                    print("Already at the first change frame.")
                continue

            if user_input.lower() in ('c', 'clear', ' clear'):
                if not prev_prompt_saved:
                    input_ids_buffer[prev_idx:idx] = prev_input_ids
                    token_type_ids_buffer[prev_idx:idx] = prev_token_type_ids
                    attention_mask_buffer[prev_idx:idx] = prev_attention_mask
                    prev_prompt_saved = True
                current_idx += 1
            elif user_input == "" or user_input == " ":
                current_idx += 1
                continue
            else:
                if not prev_prompt_saved and 'prev_input_ids' in locals():
                    input_ids_buffer[prev_idx:idx] = prev_input_ids
                    token_type_ids_buffer[prev_idx:idx] = prev_token_type_ids
                    attention_mask_buffer[prev_idx:idx] = prev_attention_mask

                tokenized = tokenizer(
                    user_input,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=32
                )
                prev_idx = idx
                prev_input_ids = tokenized['input_ids'].numpy()
                prev_token_type_ids = tokenized['token_type_ids'].numpy()
                prev_attention_mask = tokenized['attention_mask'].numpy()
                prev_prompt_saved = False
                current_idx += 1

        if not prev_prompt_saved and 'prev_input_ids' in locals():
            input_ids_buffer[change_frames[-1]:] = prev_input_ids
            token_type_ids_buffer[change_frames[-1]:] = prev_token_type_ids
            attention_mask_buffer[change_frames[-1]:] = prev_attention_mask

        if 'prompts/text' in f:
            del f['prompts/text']
        if 'observations/masks' in f:
            del f['observations/masks']
            print("Deleted 'observations/masks' dataset.")

        f.create_dataset('prompts/text/input_ids', data=input_ids_buffer, dtype='int64')
        f.create_dataset('prompts/text/token_type_ids', data=token_type_ids_buffer, dtype='int64')
        f.create_dataset('prompts/text/attention_mask', data=attention_mask_buffer, dtype='int64')

        print("Prompts saved.")

if __name__ == "__main__":
    main()
