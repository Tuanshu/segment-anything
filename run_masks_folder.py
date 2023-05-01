import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append("..")


def save_resized_image_to_subfolder(image_path, input_folder, target_size):
    original_images_folder = os.path.join(input_folder, "original_images")
    os.makedirs(original_images_folder, exist_ok=True)

    resized_image = resize_image_if_needed(image_path, target_size)
    resized_image_path = os.path.join(original_images_folder, image_path.name)

    cv2.imwrite(resized_image_path, resized_image)
    return resized_image


def get_mask_generator():
    sam_checkpoint = r"C:\Users\TuanShu\repos\segment-anything\sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def count_images(input_folder):
    return sum(1 for p in Path(input_folder).glob('*.*') if not p.stem.endswith('_mask'))


def resize_image_if_needed(image_path, target_size):
    image = cv2.imread(str(image_path))

    if image.shape[:2] != target_size[::-1]:
        print(f'the size of image is not expected, resizing to {target_size}, {image_path}')
        return cv2.resize(image, target_size)
    return image


def process_image(image_path, input_folder, predefined_size):
    # Resize the image if it doesn't match the predefined size and overwrite the original image
    image = save_resized_image_to_subfolder(image_path, input_folder, predefined_size)
    cv2.imwrite(str(image_path), image)

    # Generate masks
    masks = mask_generator.generate(image)

    # Sort the masks by area (large to small)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Create a subfolder named after the image stem
    subfolder = os.path.join(input_folder, image_path.stem)
    os.makedirs(subfolder, exist_ok=True)

    # Save the masks in the subfolder with the specified naming pattern
    for idx, mask_dict in enumerate(sorted_masks, 1):
        mask = mask_dict['segmentation']
        area = mask_dict['area']

        # Convert the boolean mask to uint8
        mask = (mask * 255).astype(np.uint8)

        mask_filename = f"{idx:03d}_{area}.jpg"
        mask_filepath = os.path.join(subfolder, mask_filename)
        cv2.imwrite(mask_filepath, mask)


def main(input_folder, predefined_size):
    num_images = count_images(input_folder)

    for image_path in tqdm((p for p in Path(input_folder).glob('*.*') if not p.stem.endswith('_mask')), total=num_images):
        process_image(image_path, input_folder, predefined_size)

    print("Masks generated and saved successfully.")


if __name__ == "__main__":
    input_folder = 'notebooks/images/hip'
    predefined_size = (512, 512)
    mask_generator = get_mask_generator()
    main(input_folder, predefined_size)
