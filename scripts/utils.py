import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import List, Optional, Union


def create_output_dir(output_dir: Union[str, Path]) -> Path:
    """Creates output directory if it doesn't exist."""
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)
    return output_dir_path


def read_image(input_name: str) -> np.ndarray:
    """Reads an image from a file."""
    return cv2.imread(input_name)


def imwrite(
    input_name: str, output_dir_pathlib: Path, hmin: Optional[int], hmax: Optional[int], images: List[np.ndarray]
) -> None:
    """Writes images horizontally concatenated into a single image file."""
    im_h = cv2.hconcat([draw_roi(img, hmin, hmax) for img in images])
    output_path = str(output_dir_pathlib.joinpath(Path(input_name).stem + "_output.jpg"))
    cv2.imwrite(output_path, im_h, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"Result image was saved at {output_path}")


def draw_roi(img: np.ndarray, hmin: Optional[int], hmax: Optional[int]) -> np.ndarray:
    """Draws a rectangle (Region of Interest) on the image."""
    pt1 = (-1, hmax)
    pt2 = (img.shape[1], hmin)
    return cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=3)


def as_3ch_grayscale_with_roi(
    gray_image: np.ndarray, shape: tuple, hmin: Optional[int], hmax: Optional[int]
) -> np.ndarray:
    """Converts a grayscale image to BGR with a rectangle (Region of Interest)."""
    image = np.zeros(shape, dtype=np.uint8)
    image[:, :, 0] = gray_image
    image[:, :, 1] = gray_image
    image[:, :, 2] = gray_image
    draw_roi(image, hmin, hmax)
    return image


def alpha_blend(
    img1: np.ndarray, img2: np.ndarray, hmin: int = 0, hmax: Optional[int] = None, alpha: float = 0.4
) -> np.ndarray:
    """Blends two images together using alpha blending."""
    img2_expand = np.zeros(img1.shape, dtype=np.uint8)
    img2_expand[hmin:hmax] = img2
    overlay_img = (alpha * img1 + (1 - alpha) * img2_expand).astype(np.uint8)
    return overlay_img


def load_image_paths(image_dir_path: str) -> List[str]:
    image_dir_pathlib = Path(image_dir_path)
    return np.sort([str(path) for path in image_dir_pathlib.glob("*") if path.suffix in [".jpg", ".png"]])


def convert_bgr2rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_image_from_ndarray(image_array: np.ndarray, figsize: Optional[tuple] = None, bgr2rgb: bool = True) -> None:
    if figsize is not None:
        plt.figure(figsize=figsize)

    if (len(image_array.shape) < 3) or (image_array.shape[-1] == 1):
        plt.imshow(image_array)
        plt.axis("off")
    else:
        if bgr2rgb:
            plt.imshow(convert_bgr2rgb(image_array))
        else:
            plt.imshow(image_array)
        plt.axis("off")


def draw_multiple_image(
    titles: List[str], images: List[np.ndarray], bgr2rgb: bool = True, figsize: Optional[tuple] = None
) -> None:
    n_images = len(images)
    assert len(titles) == n_images

    if figsize is None:
        _, axes = plt.subplots(1, n_images)
    else:
        _, axes = plt.subplots(1, n_images, figsize=figsize)
    for i in range(n_images):
        if bgr2rgb:
            axes[i].imshow(convert_bgr2rgb(images[i]))
        else:
            axes[i].imshow(images[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")
