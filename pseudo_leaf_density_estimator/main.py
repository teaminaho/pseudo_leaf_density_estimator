#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab, lab2lch
import toml
import click
import pdb

SCRIPT_DIR = Path(__file__).parent.resolve()
CONF_PATH = SCRIPT_DIR/"conf/conf.toml"
OUTPUT_DIR = SCRIPT_DIR/"data/output"
OUTPUT_DIR_RESULT = f"{OUTPUT_DIR}/result/"
OUTPUT_DIR_EXGREEN = f"{OUTPUT_DIR}/ex_green/"
OUTPUT_DIR_VALUES = f"{OUTPUT_DIR}/values/"
OUTPUT_DIR_CONCAT = f"{OUTPUT_DIR}/concat/"

with open(str(CONF_PATH), "r") as f:
    config = toml.load(f)
LSH_LOWER = config["lsh_lower"]
LSH_UPPER = config["lsh_upper"]
HSV_LOWER = config["hsv_lower"]
HSV_UPPER = config["hsv_upper"]
GRID_SIZE = config["grid_size"]
DENSITY_MIN = config["density_min"]
KERNEL_SIZE = config["kernel_size"]
NUM_DENSITY_BINS = config["num_density_bins"]
DIV_AREA = config["divided_area"]

K_H_SIZE = config["k_h_size"]
SERCH_AREA= config["serch_area"]

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
    # calculate width_size height_size depending on num_divided_width
    num_divided_width = num_divided_width
    height_size, width_size = pseudo_mask.shape
    grid_size_w = int(width_size / num_divided_width)
    num_divided_height = int(height_size / grid_size_w)
    grid_size_h = int(height_size / num_divided_height)
    # make zeros_array
    green_average = np.zeros([height_size, width_size])

    for i in range(num_divided_height):
        h_s_point = grid_size_h * i
        h_e_point = grid_size_h * (i + 1)
        divided_green_dominant = pseudo_mask[h_s_point:h_e_point, :]
        for j in range(num_divided_width):
            w_s_point = grid_size_w * j
            w_e_point = grid_size_w * (j + 1)
            green_area = np.sum(divided_green_dominant[:, w_s_point:w_e_point] == 0)
            average = green_area / (grid_size_h * grid_size_w)
            # draw the average for middle of grid
            text_point = (w_s_point + num_divided_width // 2, h_s_point + num_divided_height // 2)
            green_average[h_s_point:h_e_point, :][:, w_s_point:w_e_point] = average
    return  green_average


def convert_color(output_img, condition_img, low_condition, high_condition, value):
    img_out = output_img.copy()
    img_out[(low_condition < condition_img) & (condition_img <= high_condition)] = value
    return img_out


def gray2bgr(gray_image, shape, hmin, hmax):
    image = np.zeros(shape, dtype=np.uint8)
    image[:,:, 0] = gray_image
    image[:,:, 1] = gray_image
    image[:,:, 2] = gray_image
    draw_roi(image, hmin, hmax)
    return image


def alpha_blend(img1, img2, hmin=0, hmax=None, alpha=0.4):
    img2_expand = np.zeros(img1.shape, dtype=np.uint8)
    img2_expand[hmin:hmax] = img2
    overlay_img = (alpha * img1 + (1 - alpha) * img2_expand).astype(np.uint8)
    return overlay_img


def imwrite(input_name, hmin, hmax, images):
    im_h = cv2.hconcat([draw_roi(img, hmin, hmax) for img in images])
    print(str(OUTPUT_DIR) + Path(input_name).stem)
    cv2.imwrite(str(OUTPUT_DIR)+ "/" + Path(input_name).stem + "_output.png", im_h)


def normalize(gray_img, v_min):
    float_img = gray_img.astype(float)
    return (np.clip(float_img - v_min, 0., None) / (255. - v_min) * 255.).astype(np.uint8)


def discretize(gray_img):
    bins = np.array(DIV_AREA).astype(np.uint8)[1:]
    convert_bins = np.linspace(0, 255, NUM_DENSITY_BINS + 1).astype(np.uint8)[1:]
    out = gray_img.copy()
    v_min = -1.
    for num in range(len(bins)):
        v_max = bins[num]
        in_range = (v_min < gray_img) & (gray_img <= v_max)
        # out[in_range] = v_max
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
    h, w = mask_img.shape 
    h1, h2 = serching_range
    img_extract_crop = mask_img[h1:h2]

    # extract edge
    # define kernel
    if k_h_size % 2 != 0:
        print("hight_size is not else. so you should put else hight_size")
        sys.exit()
    else:
        kernel = np.ones((k_h_size, w))
        kernel[int(k_h_size/2):] = -1

    # 空間フィルター適用 (h,w) -> h
    img_edge = np.zeros(img_extract_crop.shape[0])
    for i in range(0, (h2-h1) - k_h_size):
        value = np.sum(img_extract_crop[i:k_h_size + i] * kernel)
        img_edge[i] = value

    # max edge index is under border line
    h_max_crop = np.argmax(img_edge)
    h_max = h_max_crop + serching_range[0]
    return h_max

@click.command()
@click.argument('input_path')
@click.option('--hmin', type=int)
@click.option('--hmax', type=int)
def main(input_path, hmin, hmax):
    print(f"input_path: {input_path}, hmin: {hmin}, hmax: {hmax}")
    # Input (original image)
    leaf_image_bgr = read_image(input_path)
    leaf_image_rgb = cv2.cvtColor(leaf_image_bgr, cv2.COLOR_BGR2RGB)
    leaf_image_lsh = calculate_perception(rgb2lch(leaf_image_rgb))
    light_mask = extract_bright_area(leaf_image_lsh)
    # decide_hmin
    if hmin == None and hmin == None:
        hmax = decide_edge(light_mask, K_H_SIZE, SERCH_AREA)
        hmin=0
    elif hmax == None:
        hmax = decide_edge(light_mask, K_H_SIZE, SERCH_AREA)
        print(f"hmin:{hmin},hmax:{hmax}")
    elif hmin == None:
        hmin = 0

    # Enable RoI
    leaf_image_bgr_crop = crop(leaf_image_bgr, hmin, hmax)
    leaf_image_rgb_crop = crop(leaf_image_rgb, hmin, hmax)

    # Binarization
    # img_lch_crop = rgb2lch(leaf_image_rgb_crop)
    # img_lsh_crop = calculate_perception(img_lch_crop)
    pseudo_mask = 255 - (extract_bright_area(leaf_image_lsh) & np.bitwise_not(extract_green_area(leaf_image_bgr)))
    pseudo_mask_crop = pseudo_mask[hmin:hmax]

    #pseudo_mask_bgr Visualization (mask)
    pseudo_mask_bgr = gray2bgr(pseudo_mask, leaf_image_bgr.shape, hmin, hmax)

    # Visualization (heatmap)
    density_img = normalize(cv2.blur(pseudo_mask_crop, (KERNEL_SIZE, KERNEL_SIZE)), v_min=DENSITY_MIN)
    heatmap_img = cv2.applyColorMap(density_img, cv2.COLORMAP_JET)
    overlay_heatmap = alpha_blend(leaf_image_bgr, heatmap_img, hmin, hmax)

    # Visualization (contour)
    contour_img = cv2.applyColorMap(discretize(density_img), cv2.COLORMAP_JET)
    overlay_contour = alpha_blend(leaf_image_bgr, contour_img, hmin, hmax)

    # Visualization (grid)
    green_average = get_gridview(pseudo_mask_crop, num_divided_width=GRID_SIZE)
    green_average_norm = normalize((1 - green_average) * 255, v_min=DENSITY_MIN)
    green_average_digi = discretize(green_average_norm)
    grid_img = cv2.applyColorMap(green_average_digi, cv2.COLORMAP_JET)
    overlay_grid = alpha_blend(leaf_image_bgr, grid_img, hmin, hmax)

    # output_all_images
    all_images_list = [leaf_image_bgr, pseudo_mask_bgr, overlay_heatmap, overlay_contour, overlay_grid]
    imwrite(input_path, hmin, hmax, all_images_list)
    #origin,2value,ikeuchidivision
    # imwrite(input_path, hmin, hmax,
    #          [leaf_image_bgr, pseudo_mask_bgr, overlay_heatmap])


if __name__ == "__main__":
    main()

