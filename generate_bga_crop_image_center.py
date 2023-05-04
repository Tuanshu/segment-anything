import numpy as np
import cv2

import json
import os
from pathlib import Path
from tqdm import tqdm


def crop_and_resize(image, point_coords, crop_size, resize_size):
    x, y = int(point_coords[0]), int(point_coords[1])
    half_crop_width, half_crop_height = crop_size[0] // 2, crop_size[1] // 2

    x_min, x_max = x - half_crop_width, x + half_crop_width
    y_min, y_max = y - half_crop_height, y + half_crop_height

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

        meta_folder = os.path.join(input_folder, image_folder.name + 'meta')
        small_images_folder = os.path.join(input_folder, image_folder.name + 'small_images')
        os.makedirs(small_images_folder, exist_ok=True)

        for json_path in Path(meta_folder).glob("*.json"):
            with open(json_path) as json_file:
                metadata = json.load(json_file)

            point_coords = metadata['point_coords'][0]
            small_image = crop_and_resize(original_image, point_coords, crop_size, resize_size)

            if small_image is None:
                continue

            small_image_filename = os.path.splitext(json_path.name)[0] + '.jpg'
            small_image_path = os.path.join(small_images_folder, small_image_filename)
            cv2.imwrite(small_image_path, small_image)


def main(input_folder, crop_size, resize_size):
    process_masks(input_folder, crop_size, resize_size)
    print("Cropped and resized images saved successfully.")


if __name__ == "__main__":
    input_folder = 'notebooks/images/hip'
    crop_size = (50, 50)
    resize_size = (32, 32)
    main(input_folder, crop_size, resize_size)
