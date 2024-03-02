#!/usr/bin/env python3
import cv2
import click
import numpy as np
from pathlib import Path
from scripts.utils import create_output_dir, read_image, imwrite, to_grayscale_with_roi, alpha_blend
from scripts.leaf_density_index import (
    rgb2lch,
    extract_bright_area,
    extract_green_area,
    calculate_perception,
    get_gridview,
    normalize,
    discretize,
    decide_edge,
)
from scripts.config import LDIConfig


SCRIPT_DIR = Path(__file__).parent.resolve()


@click.command(help="A script for processing images to calculate a pseudo leaf density index.")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--conf-path",
    "conf_path",
    default=SCRIPT_DIR.joinpath("conf", "conf.toml"),
    type=click.Path(exists=True),
    help="Path to the configuration file (in TOML format).",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=SCRIPT_DIR.joinpath("output"),
    type=click.Path(),
    help="Path to the directory where output images will be saved.",
)
@click.option("--hmin", type=int, default=None, help="Starting height for image cropping. Defaults to 0 if omitted.")
@click.option(
    "--hmax", type=int, default=None, help="Ending height for image cropping. Automatically calculated if omitted."
)
def main(input_path, conf_path, output_dir, hmin, hmax):
    print(f"input_path: {input_path}, hmin: {hmin}, hmax: {hmax}")

    # Load configuration
    config: LDIConfig = LDIConfig.from_toml(conf_path)

    # Create output directory
    output_dir_pathlib = create_output_dir(output_dir)

    # Input (original image)
    leaf_image_bgr = read_image(input_path)
    leaf_image_rgb = cv2.cvtColor(leaf_image_bgr, cv2.COLOR_BGR2RGB)
    leaf_image_lsh = calculate_perception(rgb2lch(leaf_image_rgb))
    light_mask = extract_bright_area(leaf_image_lsh, config.lsh_lower, config.lsh_upper)

    # decide_hmin
    if hmin is None:
        hmin = 0

    # decide_hmax
    if hmax is None:
        hmax = decide_edge(light_mask, config.k_h_size, config.serch_area)
    print(f"hmin:{hmin},hmax:{hmax}")

    # Binarization
    pseudo_mask = 255 - (
        extract_bright_area(leaf_image_lsh, config.lsh_lower, config.lsh_upper)
        & np.bitwise_not(extract_green_area(leaf_image_bgr, config.hsv_lower, config.hsv_upper))
    )
    pseudo_mask_crop = pseudo_mask[hmin:hmax]

    # Generate pseudo_mask_bgr Visualization (mask)
    pseudo_mask_bgr = to_grayscale_with_roi(pseudo_mask, leaf_image_bgr.shape, hmin, hmax)

    # Visualization (heatmap)
    density_img = normalize(
        cv2.blur(pseudo_mask_crop, (config.kernel_size, config.kernel_size)), v_min=config.density_min
    )
    heatmap_img = cv2.applyColorMap(density_img, cv2.COLORMAP_JET)
    overlay_heatmap = alpha_blend(leaf_image_bgr, heatmap_img, hmin, hmax)

    # Visualization (contour)
    contour_img = cv2.applyColorMap(
        discretize(density_img, config.num_density_bins, config.divided_area), cv2.COLORMAP_JET
    )
    overlay_contour = alpha_blend(leaf_image_bgr, contour_img, hmin, hmax)

    # Visualization (grid)
    green_average = get_gridview(pseudo_mask_crop, num_divided_width=config.grid_size)
    green_average_norm = normalize((1 - green_average) * 255, v_min=config.density_min)
    green_average_digi = discretize(green_average_norm, config.num_density_bins, config.divided_area)
    grid_img = cv2.applyColorMap(green_average_digi, cv2.COLORMAP_JET)
    overlay_grid = alpha_blend(leaf_image_bgr, grid_img, hmin, hmax)

    # Output all images
    all_images_list = [leaf_image_bgr, pseudo_mask_bgr, overlay_heatmap, overlay_contour, overlay_grid]
    imwrite(input_path, output_dir_pathlib, hmin, hmax, all_images_list)


if __name__ == "__main__":
    main()
