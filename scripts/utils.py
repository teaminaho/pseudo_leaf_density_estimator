import cv2
import numpy as np
from pathlib import Path


def create_output_dir(output_dir):
    """Creates output directory if it doesn't exist."""
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)
    return output_dir_path


def read_image(input_name):
    """Reads an image from a file."""
    return cv2.imread(input_name)


def imwrite(input_name, output_dir_pathlib, hmin, hmax, images):
    """Writes images horizontally concatenated into a single image file."""
    im_h = cv2.hconcat([draw_roi(img, hmin, hmax) for img in images])
    output_path = str(output_dir_pathlib.joinpath(Path(input_name).stem + "_output.jpg"))
    cv2.imwrite(output_path, im_h, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"Result image was saved at {output_path}")


def draw_roi(img, hmin, hmax):
    """Draws a rectangle (Region of Interest) on the image."""
    pt1 = (-1, hmax)
    pt2 = (img.shape[1], hmin)
    return cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3)


def as_3ch_grayscale_with_roi(gray_image, shape, hmin, hmax):
    """Converts a grayscale image to BGR with a rectangle (Region of Interest)."""
    image = np.zeros(shape, dtype=np.uint8)
    image[:, :, 0] = gray_image
    image[:, :, 1] = gray_image
    image[:, :, 2] = gray_image
    draw_roi(image, hmin, hmax)
    return image


def alpha_blend(img1, img2, hmin=0, hmax=None, alpha=0.4):
    """Blends two images together using alpha blending."""
    img2_expand = np.zeros(img1.shape, dtype=np.uint8)
    img2_expand[hmin:hmax] = img2
    overlay_img = (alpha * img1 + (1 - alpha) * img2_expand).astype(np.uint8)
    return overlay_img
