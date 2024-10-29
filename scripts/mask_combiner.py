from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np

from bsmu.vision.plugins.loaders.image.common import CommonImageFileLoader


def load_mask(mask_path: Path) -> np.ndarray:
    loader = CommonImageFileLoader()
    mask = loader.load_file(mask_path)
    return mask.pixels


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--masks-dir', type=Path, help='Path to the directory containing mask directories')
    args = arg_parser.parse_args()

    masks_dir: Path = args.masks_dir
    print(f'Processing masks directory: {masks_dir}')

    main_mask_dir_name = 'masks'
    tissue_mask_dir_name = 'masks-tissue'
    combined_mask_dir_name = 'masks-combined'
    important_regions_mask_dir_name = 'masks-important-regions'

    for main_mask_path in (masks_dir / main_mask_dir_name).iterdir():
        if not main_mask_path.is_file() or main_mask_path.suffix != '.png':
            continue

        print(f'Processing mask: {main_mask_path.name}')

        tissue_mask_path = masks_dir / tissue_mask_dir_name / main_mask_path.name
        combined_mask_path = masks_dir / combined_mask_dir_name / main_mask_path.name
        important_regions_mask_path = masks_dir / important_regions_mask_dir_name / main_mask_path.name

        main_mask = load_mask(main_mask_path)
        tissue_mask = load_mask(tissue_mask_path)

        # non_tissue_mask = np.zeros_like(tissue_mask)
        # non_tissue_mask[tissue_mask != 1] = 255

        # The `important_regions_mask` is intended to create a mask of anomalous regions
        # where non-tissue class intersect with classes that should only be present in tissue
        important_regions_mask = np.zeros_like(main_mask)
        important_regions_mask[(main_mask > 2) & (main_mask < 8) & (tissue_mask != 1)] = 1

        # Dilate the important regions to enhance visibility
        dilation_kernel_size = 11
        dilation_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        cv.dilate(important_regions_mask, dilation_kernel, dst=important_regions_mask, iterations=4)

        # Update the main mask to label non-tissue areas
        main_mask[tissue_mask != 1] = 255

        cv.imwrite(str(combined_mask_path), main_mask)
        cv.imwrite(str(important_regions_mask_path), important_regions_mask)


if __name__ == '__main__':
    main()
