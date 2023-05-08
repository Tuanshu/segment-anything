import numpy as np
import cv2

import json
import os
from pathlib import Path
from tqdm import tqdm


def crop_and_resize_bbox(image, bbox, resize_size):
    x_min, y_min, x_max, y_max = bbox

    if x_min < 0 or x_max > image.shape[1] or y_min < 0 or y_max > image.shape[0]:
        return None

    cropped = image[y_min:y_max, x_min:x_max]
    resized = cv2.resize(cropped, resize_size)

    return resized


def process_masks(input_folder, crop_size, resize_size):
    image_folders = [p for p in Path(input_folder).iterdir() if p.is_dir() and not p.name.endswith('meta')]

    for image_folder in tqdm(image_folders):
        original_image_path = os.path.join(input_folder, "original_images", f"{image_folder.name}.jpg")
        original_image = cv2.imread(original_image_path)

        red_mask_image_path = os.path.join(input_folder, "red_masks", f"{image_folder.name}_mask.jpg")
        red_mask_image = cv2.imread(red_mask_image_path, cv2.IMREAD_GRAYSCALE)

        meta_folder = os.path.join(input_folder, image_folder.name + 'meta')
        small_images_folder_255 = os.path.join(input_folder, image_folder.name + '_small_images_255')
        small_images_folder_0 = os.path.join(input_folder, image_folder.name + '_small_images_0')

        for json_path in Path(meta_folder).glob("*.json"):
            with open(json_path) as json_file:
                metadata = json.load(json_file)

            if metadata['area'] > 400:
                continue

            if metadata['area'] < 250:
                continue

            x_min, y_min, width, height = metadata['bbox']
            x_max, y_max = x_min + width, y_min + height
            bbox = [x_min, y_min, x_max, y_max]
            small_image = crop_and_resize_bbox(original_image, bbox, resize_size)

            if small_image is None:
                continue

            # Check the center coordinate of the bounding box in the red_mask image
            center_x = x_min + width // 2
            center_y = y_min + height // 2
            pixel_value = red_mask_image[center_y, center_x]

            small_image_filename = os.path.splitext(json_path.name)[0] + '.jpg'

            if pixel_value == 255:
                os.makedirs(small_images_folder_255, exist_ok=True)

                small_image_path = os.path.join(small_images_folder_255, small_image_filename)
            else:
                os.makedirs(small_images_folder_0, exist_ok=True)

                small_image_path = os.path.join(small_images_folder_0, small_image_filename)

            cv2.imwrite(small_image_path, small_image)


def main(input_folder, crop_size, resize_size):
    process_masks(input_folder, crop_size, resize_size)
    print("Cropped, classified, and resized images saved successfully.")


if __name__ == "__main__":
    input_folder = 'notebooks/images/hip'
    crop_size = (50, 50)
    resize_size = (32, 32)
    main(input_folder, crop_size, resize_size)
