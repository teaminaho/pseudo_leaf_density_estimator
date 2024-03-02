#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab, lab2lch
from skimage.measure import block_reduce
import toml
import click

SCRIPT_DIR = Path(__file__).parent.resolve()


def create_output_dir(output_dir):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)
    return output_dir_path


def rgb2lch(rgbimg):
    img_lab = rgb2lab(rgbimg)
    return lab2lch(img_lab)


def read_image(input_name):
    return cv2.imread(input_name)


def extract_bright_area(img_lsh, lsh_lower, lsh_upper):
    return cv2.inRange(img_lsh, np.array(lsh_lower), np.array(lsh_upper))


def extract_green_area(img_bgr, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))


def calculate_perception(img_lch):
    img_new = img_lch.copy()
    img_new[:, :, 1] = (255 * np.log10(img_new[:, :, 1] + 1) / np.log10(255)).astype(np.uint8)
    return img_new


def get_gridview(pseudo_mask, num_divided_width=4):
    green_average = pseudo_mask.copy()
    height_size, width_size = pseudo_mask.shape
    grid_size_w = int(width_size / num_divided_width)
    num_divided_height = int(height_size / grid_size_w)
    grid_size_h = int(height_size / num_divided_height)
    green_average = block_reduce(green_average, block_size=(grid_size_h, grid_size_w), func=np.mean)
    green_average_expand = cv2.resize(green_average, dsize=(width_size, height_size), interpolation=cv2.INTER_NEAREST)

    return (255 - green_average_expand) / 255


def convert_color(output_img, condition_img, low_condition, high_condition, value):
    img_out = output_img.copy()
    img_out[(low_condition < condition_img) & (condition_img <= high_condition)] = value
    return img_out


def gray2bgr(gray_image, shape, hmin, hmax):
    image = np.zeros(shape, dtype=np.uint8)
    image[:, :, 0] = gray_image
    image[:, :, 1] = gray_image
    image[:, :, 2] = gray_image
    draw_roi(image, hmin, hmax)
    return image


def alpha_blend(img1, img2, hmin=0, hmax=None, alpha=0.4):
    img2_expand = np.zeros(img1.shape, dtype=np.uint8)
    img2_expand[hmin:hmax] = img2
    overlay_img = (alpha * img1 + (1 - alpha) * img2_expand).astype(np.uint8)
    return overlay_img


def imwrite(input_name, output_dir_pathlib, hmin, hmax, images):
    im_h = cv2.hconcat([draw_roi(img, hmin, hmax) for img in images])
    output_path = str(output_dir_pathlib.joinpath(Path(input_name).stem + "_output.png"))
    cv2.imwrite(output_path, im_h)
    print(f"Result image was saved at {output_path}")


def normalize(gray_img, v_min):
    float_img = gray_img.astype(float)
    return (np.clip(float_img - v_min, 0.0, None) / (255.0 - v_min) * 255.0).astype(np.uint8)


def discretize(gray_img, num_density_bins, div_area):
    bins = np.array(div_area).astype(np.uint8)[1:]
    convert_bins = np.linspace(0, 255, num_density_bins + 1).astype(np.uint8)[1:]
    out = gray_img.copy()
    v_min = -1.0
    for num in range(len(bins)):
        v_max = bins[num]
        in_range = (v_min < gray_img) & (gray_img <= v_max)
        out[in_range] = convert_bins[num]
        v_min = v_max
    return out


def draw_roi(img, hmin, hmax):
    pt1 = (-1, hmax)
    pt2 = (img.shape[1], hmin)
    return cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3)


def crop(img, hmin, hmax):
    return img[hmin:hmax, :, :]


def decide_edge(mask_img, k_h_size, serching_range):
    _, w = mask_img.shape
    h1, h2 = serching_range
    img_extract_crop = mask_img[h1:h2]

    # Edge extraction

    # define kernel
    if k_h_size % 2 != 0:
        print("hight_size is not else. so you should put else hight_size")
        sys.exit()
    else:
        kernel = np.ones((k_h_size, w))
        kernel[int(k_h_size / 2) :] = -1

    # Spatial filter application (h,w) -> h
    img_edge = np.zeros(img_extract_crop.shape[0])
    for i in range(0, (h2 - h1) - k_h_size):
        value = np.sum(img_extract_crop[i : k_h_size + i] * kernel)
        img_edge[i] = value

    # max edge index is under border line
    h_max_crop = np.argmax(img_edge)
    h_max = h_max_crop + serching_range[0]
    return h_max


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
    with open(str(conf_path), "r") as f:
        config = toml.load(f)

    # Create output directory
    output_dir_pathlib = create_output_dir(output_dir)

    # Input (original image)
    leaf_image_bgr = read_image(input_path)
    leaf_image_rgb = cv2.cvtColor(leaf_image_bgr, cv2.COLOR_BGR2RGB)
    leaf_image_lsh = calculate_perception(rgb2lch(leaf_image_rgb))
    light_mask = extract_bright_area(leaf_image_lsh, config["lsh_lower"], config["lsh_upper"])

    # decide_hmin
    if hmin is None:
        hmin = 0

    # decide_hmax
    if hmax is None:
        hmax = decide_edge(light_mask, config["k_h_size"], config["serch_area"])
    print(f"hmin:{hmin},hmax:{hmax}")

    # Binarization
    pseudo_mask = 255 - (
        extract_bright_area(leaf_image_lsh, config["lsh_lower"], config["lsh_upper"])
        & np.bitwise_not(extract_green_area(leaf_image_bgr, config["hsv_lower"], config["hsv_upper"]))
    )
    pseudo_mask_crop = pseudo_mask[hmin:hmax]

    # Generate pseudo_mask_bgr Visualization (mask)
    pseudo_mask_bgr = gray2bgr(pseudo_mask, leaf_image_bgr.shape, hmin, hmax)

    # Visualization (heatmap)
    density_img = normalize(
        cv2.blur(pseudo_mask_crop, (config["kernel_size"], config["kernel_size"])), v_min=config["density_min"]
    )
    heatmap_img = cv2.applyColorMap(density_img, cv2.COLORMAP_JET)
    overlay_heatmap = alpha_blend(leaf_image_bgr, heatmap_img, hmin, hmax)

    # Visualization (contour)
    contour_img = cv2.applyColorMap(
        discretize(density_img, config["num_density_bins"], config["divided_area"]), cv2.COLORMAP_JET
    )
    overlay_contour = alpha_blend(leaf_image_bgr, contour_img, hmin, hmax)

    # Visualization (grid)
    green_average = get_gridview(pseudo_mask_crop, num_divided_width=config["grid_size"])
    green_average_norm = normalize((1 - green_average) * 255, v_min=config["density_min"])
    green_average_digi = discretize(green_average_norm, config["num_density_bins"], config["divided_area"])
    grid_img = cv2.applyColorMap(green_average_digi, cv2.COLORMAP_JET)
    overlay_grid = alpha_blend(leaf_image_bgr, grid_img, hmin, hmax)

    # Output all images
    all_images_list = [leaf_image_bgr, pseudo_mask_bgr, overlay_heatmap, overlay_contour, overlay_grid]
    imwrite(input_path, output_dir_pathlib, hmin, hmax, all_images_list)


if __name__ == "__main__":
    main()
