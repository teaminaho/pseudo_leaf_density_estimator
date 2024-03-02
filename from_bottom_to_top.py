#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab, lab2lch
from skimage.measure import block_reduce
import toml
import click

SCRIPT_DIR = Path(__file__).parent.resolve()
CONF_PATH = SCRIPT_DIR / "conf/conf_bottom.toml"
OUTPUT_DIR = SCRIPT_DIR / "data/output"

with open(str(CONF_PATH), "r") as f:
    config = toml.load(f)
LSH_LOWER = config["lsh_lower"]
LSH_UPPER = config["lsh_upper"]
HSV_LOWER = config["hsv_lower"]
HSV_UPPER = config["hsv_upper"]
CROP_RATE = config["crop_rate"]
BORDER_COLOR = config["border_color"]


def rgb2lch(rgbimg):
    img_lab = rgb2lab(rgbimg)
    return lab2lch(img_lab)


def read_image(input_name):
    return cv2.imread(input_name)


def extract_bright_area(img_lsh):
    return cv2.inRange(img_lsh, np.array(LSH_LOWER), np.array(LSH_UPPER))


def extract_green_area(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))


def calculate_perception(img_lch):
    img_new = img_lch.copy()
    img_new[:, :, 1] = (255 * np.log10(img_new[:, :, 1] + 1) / np.log10(255)).astype(np.uint8)
    return img_new


def get_gridview(pseudo_mask, num_divided_width=4):
    green_average = pseudo_mask.copy()
    h_size, w_size = pseudo_mask.shape
    block = (w_size // num_divided_width) + 1
    w_pad = block - (w_size % (block))
    h_pad = block - (h_size % block)
    green_average = np.pad(green_average, ([0, h_pad], [0, w_pad]), mode="edge")
    green_average = block_reduce(green_average, block_size=(block, block), func=np.mean)
    green_average_expand = cv2.resize(green_average, dsize=(w_size, h_size), interpolation=cv2.INTER_NEAREST)
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


def normalize(gray_img, v_min):
    float_img = gray_img.astype(float)
    return (np.clip(float_img - v_min, 0.0, None) / (255.0 - v_min) * 255.0).astype(np.uint8)


def discretize(gray_img):
    bins = np.array(DIV_AREA).astype(np.uint8)[1:]
    convert_bins = np.linspace(0, 255, NUM_DENSITY_BINS + 1).astype(np.uint8)[1:]
    out = gray_img.copy()
    v_min = -1.0
    for num in range(len(bins)):
        v_max = bins[num]
        in_range = (v_min < gray_img) & (gray_img <= v_max)
        out[in_range] = convert_bins[num]
        v_min = v_max
    return out


def calc_pseurate(border_draw_img, pseudo_mask, crop_rate: int, color):
    h, w, _ = pseudo_mask.shape
    pseudo_mask4crop = pseudo_mask.copy()
    mid_p = list(map(lambda n: int(n / 2), (w, h)))
    sph = mid_p[1] - int(h / crop_rate / 2)
    eph = mid_p[1] + int(h / crop_rate / 2)
    crop_img = pseudo_mask4crop[sph:eph]
    rate = np.sum(crop_img == 0) / (len(crop_img.reshape(-1)))
    border_draw_img = cv2.rectangle(border_draw_img, (0, sph), (w, eph), color=color, thickness=3)
    pseudo_mask = cv2.rectangle(pseudo_mask, (0, sph), (w, eph), color=color, thickness=3)
    return rate, border_draw_img, pseudo_mask


@click.command()
@click.argument("input_path")
def main(input_path):
    # pathlib and var = input_path.parents
    p_file = Path(input_path)
    dir_name = p_file.parent.name
    output_dir = str(OUTPUT_DIR) + "/" + dir_name
    print(output_dir)

    # Input (original image)
    leaf_image_bgr = read_image(input_path)
    leaf_image_rgb = cv2.cvtColor(leaf_image_bgr, cv2.COLOR_BGR2RGB)
    leaf_image_lsh = calculate_perception(rgb2lch(leaf_image_rgb))
    light_mask = extract_bright_area(leaf_image_lsh)
    pseudo_mask = extract_bright_area(leaf_image_lsh) & np.bitwise_not(extract_green_area(leaf_image_bgr))
    pseudo_mask_rgb = np.zeros(leaf_image_bgr.shape, dtype=np.uint8)
    pseudo_mask_rgb[:, :, 0] = pseudo_mask
    pseudo_mask_rgb[:, :, 1] = pseudo_mask
    pseudo_mask_rgb[:, :, 2] = pseudo_mask

    border_draw_img = leaf_image_bgr.copy()
    rate_list = []

    for i in range(len(CROP_RATE)):
        rate, border_draw_img, pseudo_mask_rgb = calc_pseurate(
            border_draw_img, pseudo_mask_rgb, CROP_RATE[i], BORDER_COLOR[i]
        )
        rate_list.append(rate)
    print(f"{p_file.name}")

    with open(f"pseu_rate_bottom.csv", "a", encoding="UTF-8") as f:
        f.write(f"{output_dir},{p_file.name},{rate_list[0]},{rate_list[1]},{rate_list[2]},{rate_list[3]}\n")

    img_h = cv2.hconcat([border_draw_img, pseudo_mask_rgb])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cv2.imwrite(f"{output_dir}/{p_file.name}_rate_img.jpg", img_h)


if __name__ == "__main__":
    main()
