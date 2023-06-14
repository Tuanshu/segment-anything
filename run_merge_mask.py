import numpy as np
import cv2

import os
import re
from pathlib import Path


def detect_red_dots(image, threshold=200, upper_limit=50):
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]
    mask = (red_channel > threshold) & (blue_channel < upper_limit) & (green_channel < upper_limit)
    return mask.astype(np.uint8) * 255


def process_subfolder_red_dots(subfolder, min_area, max_area, parent_folder):
    merged_mask = None
    original_image_path = os.path.join(parent_folder, f"{subfolder.name}.jpg")
    original_image = cv2.imread(original_image_path)
    if original_image is not None:
        red_dots_mask = detect_red_dots(original_image)

        for mask_path in subfolder.glob('*.jpg'):
            match = re.search(r'\d+_(\d+).jpg', str(mask_path.name))
            if match:
                area = int(match.group(1))

                if min_area <= area <= max_area:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    overlap_mask = cv2.bitwise_and(mask, red_dots_mask)

                    if np.any(overlap_mask):
                        if merged_mask is None:
                            merged_mask = mask
                        else:
                            merged_mask = cv2.bitwise_or(merged_mask, mask)

    return merged_mask


def process_subfolder(subfolder, min_area, max_area):
    merged_mask = None

    for mask_path in subfolder.glob('*.jpg'):
        match = re.search(r'\d+_(\d+).jpg', str(mask_path.name))
        if match:
            area = int(match.group(1))

            if min_area <= area <= max_area:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                if merged_mask is None:
                    merged_mask = mask
                else:
                    merged_mask = cv2.bitwise_or(merged_mask, mask)

    return merged_mask


def save_merged_mask(merged_mask, subfolder, parent_folder):
    os.makedirs(parent_folder, exist_ok=True)

    merged_mask_filename = f"{subfolder.name}_mask.jpg"
    merged_mask_filepath = os.path.join(parent_folder, merged_mask_filename)
    cv2.imwrite(merged_mask_filepath, merged_mask)


def main(parent_folder, min_area, max_area):
    for subfolder in Path(parent_folder).glob('*'):
        if not subfolder.is_dir():
            continue

        merged_mask = process_subfolder(subfolder, min_area, max_area)
        if merged_mask is not None:
            save_merged_mask(merged_mask, subfolder, os.path.join(parent_folder, "merged_masks"))

        merged_red_mask = process_subfolder_red_dots(subfolder, min_area, max_area, parent_folder)
        if merged_red_mask is not None:
            save_merged_mask(merged_red_mask, subfolder, os.path.join(parent_folder, 'red_masks'))

    print("Merged masks and red masks generated and saved successfully.")


if __name__ == "__main__":
    parent_folder = 'notebooks/images/pih'
    parent_folder = r'C:\Users\TuanShu\repos\mxi_playground'

    min_area = 20#50
    max_area = 500#500
    main(parent_folder, min_area, max_area)
