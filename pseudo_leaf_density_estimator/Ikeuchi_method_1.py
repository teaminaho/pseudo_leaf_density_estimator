#!/usr/bin/env python3
import cv2
import sys
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab, lab2lch
import toml

SCRIPT_DIR = Path(__file__).parent.resolve()
CONF_PATH = SCRIPT_DIR / "conf/conf.toml"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR_RESULT = f"{OUTPUT_DIR}/result/"
OUTPUT_DIR_EXGREEN = f"{OUTPUT_DIR}/ex_green/"
OUTPUT_DIR_VALUES = f"{OUTPUT_DIR}/values/"
OUTPUT_DIR_CONCAT = f"{OUTPUT_DIR}/concat/"

with open(str(CONF_PATH), "r") as f:
    config = toml.load(f)
RGB_COLORMAP = config["rgb_colormap"]
LSH_LOWER = config["lsh_lower"]
LSH_UPPER = config["lsh_upper"]
GRID_SIZE = config["grid_size"]
KERNEL_SIZE = 401 #new


def rgb2lch(rgbimg):
    img_lab = rgb2lab(rgbimg)
    return lab2lch(img_lab)


def read_image(input_name):
    leaf_image = cv2.imread(input_name)
    leaf_image = cv2.resize(leaf_image, None, fx=1, fy=1)
    return leaf_image


def extract_blank_area(img_lsh, leaf_image_bgr):
    lshLower = np.array([50, 0, 0])
    lshUpper = np.array([100, 80, 6])
    lsh_mask = cv2.inRange(img_lsh, lshLower, lshUpper)
    lsh_mask = cv2.resize(lsh_mask, leaf_image_bgr.shape[-2::-1])
    return lsh_mask


def calculate_perception(img_lch):
    img_new = img_lch.copy()
    img_new[:, :, 1] = (255 * np.log10(img_new[:, :, 1] + 1) / np.log10(255)).astype(np.uint8)
    return img_new


def get_gridview(lsh_mask, num_divided_width=4):
    num_divided_width = num_divided_width
    height_size, width_size = lsh_mask.shape
    grid_size = int(width_size/num_divided_width)
    num_divided_height = int(height_size/grid_size)
    overgrow_average = np.zeros([num_divided_height,num_divided_width])
    green_average = np.zeros([height_size,width_size])
    green_average_values = np.zeros([height_size, width_size])
    for i in range(num_divided_height):
        h_s_point = grid_size*i
        h_e_point = grid_size*(i+1)
        divided_green_dominant = lsh_mask[h_s_point:h_e_point,:]
        for j in range(num_divided_width):
            w_s_point = grid_size*j
            w_e_point = grid_size*(j+1)
            green_num = np.sum(divided_green_dominant[:,w_s_point:w_e_point] == 0)
            average = green_num / (grid_size ** 2)
            text_point = (w_s_point+num_divided_width//2, h_s_point+num_divided_height//2)
            cv2.putText(green_average_values,
                        text=f"{average*100:.1f}",
                        org=text_point,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color= (255, 255, 255) if average > 0.2 else (0,0,0),
                        thickness=1,
                        lineType=cv2.LINE_4)
            overgrow_average[i,j] = average
            green_average[h_s_point:h_e_point,:][:,w_s_point:w_e_point] = average
    return overgrow_average, green_average, green_average_values


def convert_color(output_img, condition_img, low_condition, high_condition, value):
    img_out = output_img.copy()
    img_out[(low_condition < condition_img) & (condition_img <= high_condition)] = value
    return img_out


def make_color_map_3(shape, green_average):
    green_average_out = np.zeros(shape)
    value_area = [0,0.85,0.99,1.0]
    for i in range(len(RGB_COLORMAP)):
        print(i,value_area[i], value_area[i+1])
        green_average_out = convert_color(green_average_out, green_average, value_area[i], value_area[i+1], RGB_COLORMAP[i])
    return green_average_out


def make_color_map_5(shape, green_average):
    green_average_out = np.zeros(shape)
    value_area = [0,0.7,0.85,0.9,0.99,1.0]
    for i in range(len(RGB_COLORMAP)):
        print(i,value_area[i], value_area[i+1])
        green_average_out = convert_color(green_average_out, green_average, value_area[i], value_area[i+1], RGB_COLORMAP[i])
    return green_average_out


def make_color_map_10(shape, green_average):
    green_average_out = np.zeros(shape)
    value_area = [0,0.2,0.4,0.5,0.7,0.8,0.85,0.9,0.95,0.99,1.0]
    for i in range(len(RGB_COLORMAP)):
        print(i,value_area[i], value_area[i+1])
        green_average_out = convert_color(green_average_out, green_average, value_area[i], value_area[i+1], RGB_COLORMAP[i])
    return green_average_out


def alpha_blend(img1, img2, alpha=0.4):
    overlay_img = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
    return overlay_img


def imwrite(input_name, overlay_img, lsh_mask, green_average_values, leaf_image_bgr,
            heatmap_img, contour_img):
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUTPUT_DIR_EXGREEN + Path(input_name).stem + "_output.png"), lsh_mask)
    cv2.imwrite(str(OUTPUT_DIR_VALUES + Path(input_name).stem + "_output.png"), green_average_values)
    cv2.imwrite(str(OUTPUT_DIR_RESULT + Path(input_name).stem + "_output.png"), overlay_img)
    cv2.imwrite(str(OUTPUT_DIR_RESULT + Path(input_name).stem + "_output.png"), overlay_img)
    cv2.imwrite(str(OUTPUT_DIR_RESULT + Path(input_name).stem + "_output.png"), overlay_img)
    lsh_mask = np.concatenate([lsh_mask[:, :, None], lsh_mask[:, :, None], lsh_mask[:, :, None]], 2)
    im_h = cv2.hconcat([leaf_image_bgr, lsh_mask, overlay_img, heatmap_img, contour_img])
    cv2.imwrite(str(OUTPUT_DIR_CONCAT + Path(input_name).stem + "_output.png"), im_h)


def normalize(gray_img, v_min):
    float_img = gray_img.astype(float)
    return (np.clip(float_img - v_min, 0., None) / (255. - v_min) * 255.).astype(np.uint8)


def discretize(gray_img, bins):
    out = gray_img.copy()
    v_min = -1.
    for v_max in bins:
        in_range = (v_min < gray_img) & (gray_img <= v_max)
        out[in_range] = v_max
        v_min = v_max
    return out


def main(args):
    input_name = args[1] if len(args) > 1 else "data/leaf.jpg"
    leaf_image_bgr = read_image(input_name)
    leaf_image_rgb = cv2.cvtColor(leaf_image_bgr, cv2.COLOR_BGR2RGB)
    img_lch = rgb2lch(leaf_image_rgb)
    img_lsh = calculate_perception(img_lch)
    lsh_mask = extract_blank_area(img_lsh, leaf_image_bgr)
    if len(RGB_COLORMAP) == 3:
        make_color_map = make_color_map_3
    elif len(RGB_COLORMAP) == 5:
        make_color_map = make_color_map_5
    elif len(RGB_COLORMAP) == 10:
        make_color_map = make_color_map_10
    overgrow_average, green_average, green_average_values = \
        get_gridview(lsh_mask, num_divided_width=GRID_SIZE)
    green_average_rgb = make_color_map(leaf_image_rgb.shape, green_average)
    overlay_img = alpha_blend(leaf_image_rgb, green_average_rgb)
    # new
    blur_img = normalize(cv2.blur(255 - lsh_mask, (KERNEL_SIZE, KERNEL_SIZE)), v_min=220)
    digi_img = discretize(blur_img, bins=[55, 105, 155, 205, 255])
    heatmap_img = cv2.applyColorMap(blur_img, cv2.COLORMAP_JET)
    contour_img = cv2.applyColorMap(digi_img, cv2.COLORMAP_JET)
    overlay_heatmap = alpha_blend(leaf_image_rgb, heatmap_img)
    overlay_contour = alpha_blend(leaf_image_rgb, contour_img)

    imwrite(input_name, overlay_img, lsh_mask, green_average_values, leaf_image_bgr,
            overlay_heatmap, overlay_contour)


if __name__ == "__main__":
    main(sys.argv)
