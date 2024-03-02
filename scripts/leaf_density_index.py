import sys
import cv2
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab, lab2lch
from skimage.measure import block_reduce

SCRIPT_DIR = Path(__file__).parent.resolve()


def rgb2lch(rgbimg):
    """Converts an RGB image to LCH color space."""
    img_lab = rgb2lab(rgbimg)
    return lab2lch(img_lab)


def extract_bright_area(img_lsh, lsh_lower, lsh_upper):
    """Extracts bright areas based on LSH color space thresholds."""
    return cv2.inRange(img_lsh, np.array(lsh_lower), np.array(lsh_upper))


def extract_green_area(img_bgr, hsv_lower, hsv_upper):
    """Extracts green areas based on HSV color space thresholds."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))


def calculate_perception(img_lch):
    """Enhances perception of LCH image."""
    img_new = img_lch.copy()
    img_new[:, :, 1] = (255 * np.log10(img_new[:, :, 1] + 1) / np.log10(255)).astype(np.uint8)
    return img_new


def get_gridview(pseudo_mask, num_divided_width=4):
    """Generates a grid view representation of the pseudo mask."""
    green_average = block_reduce(
        pseudo_mask,
        block_size=(pseudo_mask.shape[0] // num_divided_width, pseudo_mask.shape[1] // num_divided_width),
        func=np.mean,
    )
    green_average_expand = cv2.resize(
        green_average, (pseudo_mask.shape[1], pseudo_mask.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return (255 - green_average_expand) / 255


def normalize(gray_img, v_min):
    """Normalizes a grayscale image based on a minimum value."""
    float_img = gray_img.astype(float)
    return (np.clip(float_img - v_min, 0.0, None) / (255.0 - v_min) * 255.0).astype(np.uint8)


def discretize(gray_img, num_density_bins, div_area):
    """Discretizes a grayscale image into specified number of density bins."""
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


def convert_color(output_img, condition_img, low_condition, high_condition, value):
    """Converts color of the output image based on the condition."""
    img_out = output_img.copy()
    img_out[(low_condition < condition_img) & (condition_img <= high_condition)] = value
    return img_out


def decide_edge(mask_img, k_h_size, serching_range):
    """Decides the edge based on mask image and kernel size."""
    _, w = mask_img.shape
    h1, h2 = serching_range
    img_extract_crop = mask_img[h1:h2]

    if k_h_size % 2 != 0:
        print("Kernel height size is not even. Please use an even height size.")
        sys.exit()
    else:
        kernel = np.ones((k_h_size, w))
        kernel[int(k_h_size / 2) :] = -1

    img_edge = np.zeros(img_extract_crop.shape[0])
    for i in range(0, (h2 - h1) - k_h_size):
        value = np.sum(img_extract_crop[i : k_h_size + i] * kernel)
        img_edge[i] = value

    h_max_crop = np.argmax(img_edge)
    h_max = h_max_crop + serching_range[0]
    return h_max
