import sys
import cv2
import numpy as np
from typing import List
from pathlib import Path
from skimage.color import rgb2lab, lab2lch
from skimage.measure import block_reduce

SCRIPT_DIR = Path(__file__).parent.resolve()


def rgb2lch(rgbimg):
    """Converts an RGB image to LCH color space."""
    img_lab = rgb2lab(rgbimg)
    return lab2lch(img_lab)


def extract_bright_area(img_lsh, lsh_lower=[20.0, 0.0, 0.0], lsh_upper=[100.0, 120.0, 7.0]):
    """Extracts bright areas based on LSH color space thresholds."""
    return cv2.inRange(img_lsh, np.array(lsh_lower), np.array(lsh_upper))


def extract_green_area(img_bgr, hsv_lower=[35, 40, 50], hsv_upper=[80, 255, 255]):
    """Extracts green areas based on HSV color space thresholds."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))


def calculate_perception(img_lch):
    """
    Enhances the perceptual contrast of an image in LCH color space by applying a logarithmic transformation to the Chroma component.

    This method aims to mimic the logarithmic response of human vision to color and light,
    thereby improving the visual contrast and vividness of colors in the image.
    Low chroma values are made more distinct, while high chroma values are adjusted more subtly, enhancing overall image detail and color differentiation.
    """
    img_new = img_lch.copy()

    # Apply logarithmic scaling to Chroma to enhance contrast
    # the formula adjusts Chroma based on human perceptual response to color saturation.
    img_new[:, :, 1] = (255 * np.log10(img_new[:, :, 1] + 1) / np.log10(255)).astype(np.uint8)
    return img_new


def get_gridview(pseudo_mask, num_divided_width=4):
    """Generates a grid view representation of the pseudo mask.

    This function divides the image into uniform blocks, calculates the mean
    value of each block, and resizes the reduced image back to the original
    size for visualization.

    The block size is calculated to ensure coverage of the entire image,
    adding 1 to include any remainder not covered by the division.
    Padding is added if the image size is not divisible by the block size,
    but only as much as necessary to make the division even.
    """
    h_size, w_size = pseudo_mask.shape

    # Calculate block size; +1 ensures coverage of the entire image
    block_size = (w_size // num_divided_width) + 1

    # Calculate padding to make the image divisible by block size, add padding only if necessary
    pad_h = (block_size - h_size % block_size) % block_size
    pad_w = (block_size - w_size % block_size) % block_size

    # Apply padding with mode 'edge' to avoid introducing artificial edges
    padded_mask = np.pad(pseudo_mask, ((0, pad_h), (0, pad_w)), mode="edge")

    # Reduce image size by calculating the mean value of each block
    green_average = block_reduce(padded_mask, block_size=(block_size, block_size), func=np.mean)

    # Resize the reduced image back to the original size for visualization
    green_average_expand = cv2.resize(green_average, (w_size, h_size), interpolation=cv2.INTER_NEAREST)

    # Return the grid view representation
    return (255 - green_average_expand) / 255


def normalize(gray_img, v_min):
    """Normalizes a grayscale image based on a minimum value."""
    float_img = gray_img.astype(float)
    return (np.clip(float_img - v_min, 0.0, None) / (255.0 - v_min) * 255.0).astype(np.uint8)


def discretize(gray_img, num_density_bins: int = 4, div_area: List[int] = [0, 30, 170, 240, 255]):
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
