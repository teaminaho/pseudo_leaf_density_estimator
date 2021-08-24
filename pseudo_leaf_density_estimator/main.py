import cv2
import sys
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib
import matplotlib.pyplot as plt
from guided_filter.core.filter import GuidedFilter
from skimage.color import rgb2lab, lab2lch
from pathlib import Path

# matplotlib.use('TkAgg')


def show(image):
    while not (cv2.waitKey(10) & 0xFF in [ord("q"), 27]):
        cv2.imshow("title", image)
    cv2.destroyAllWindows()


def get_density(patch_flattened):
    return np.sum(patch_flattened) / len(patch_flattened)


def main(args):
    input_name = args[1] if len(args) > 1 else "data/leaf.jpg"
    leaf_image = cv2.imread(input_name)
    # leaf_image = cv2.imread("data/leaf2.png")
    leaf_image = cv2.resize(leaf_image, None, fx=0.2, fy=0.2)

    leaf_image_raw = leaf_image
    # leaf_image_hsv = cv2.cvtColor(leaf_image, cv2.COLOR_RGB2HSV)
    leaf_image_rgb = cv2.cvtColor(leaf_image, cv2.COLOR_BGR2RGB)
    img_lab = rgb2lab(leaf_image_rgb)
    img_lch = lab2lch(img_lab)
    # show(leaf_image)

    """
    Extract green-color dominant area
    """
    bgrLower = np.array([0, 110, 0])
    bgrUpper = np.array([255, 170, 255])
    img_lch[:, :, 1] = (255 * np.log10(img_lch[:, :, 1] + 1) / np.log10(255)).astype(np.uint8)  # perceptual color spaces
    lsh_mask = cv2.inRange(img_lch, bgrLower, bgrUpper)
    lsh_mask = cv2.resize(lsh_mask, leaf_image.shape[-2::-1])
    # plt.hist(img_lch[:,:,1].reshape(-1))
    # plt.savefig("lch_b.png")
    # show(lsh_mask)

    leaf_image_float = leaf_image.astype(np.float) / 255
    leaf_image_float += 10e-8
    lsh_where = np.where(lsh_mask == 0)
    leaf_image_float[lsh_where[0], lsh_where[1], :] = (1.0, 0.1, 1.0)
    is_green_dominant = (np.max(leaf_image_float[:, ..., :] / leaf_image_float[:, :, 1][:, :, np.newaxis], axis=2) > 1.0) & (
        np.max(leaf_image_float, axis=2) > 0.5
    )
    mask_green_dominant = np.zeros_like(is_green_dominant, np.uint8)
    mask_green_dominant[is_green_dominant] = 255
    # show(mask_green_dominant)

    """
    Get pseudo-density of green components per local patches
    """    
    patch_size = 7
    patches = extract_patches_2d(mask_green_dominant, (patch_size, patch_size))
    size_restored = np.array(mask_green_dominant.shape) - [patch_size - 1, patch_size - 1]
    mask_green_dominant = (np.max(patches.reshape((-1, patch_size ** 2)), axis=-1)).reshape(size_restored)
    mask_green_dominant = mask_green_dominant.astype(np.uint8)

    green_relative_intensity = np.max(leaf_image_float[:, :, 1][:, :, np.newaxis] / leaf_image_float[:, ..., :], axis=2)
    patches_gri = extract_patches_2d(green_relative_intensity, (patch_size, patch_size)).reshape(-1, patch_size ** 2)
    mask_gri = np.apply_along_axis(get_density, 1, patches_gri).reshape(mask_green_dominant.shape)
    # gri_quant_l = np.quantile(mask_gri.reshape(-1), 0.50)
    # gri_quant_u = np.quantile(mask_gri.reshape(-1), 0.90)
    gri_quant_l = 1
    gri_quant_u = 30000
    mask_gri[mask_gri < gri_quant_l] = gri_quant_l
    mask_gri[mask_gri >= gri_quant_u] = gri_quant_u
    mask_gri_shifted = mask_gri - gri_quant_l
    mask_gri_normalized = (mask_gri_shifted - mask_gri_shifted.min()) / (mask_gri_shifted.max() - mask_gri_shifted.min())
    mask_gri_normalized = cv2.copyMakeBorder(
        mask_gri_normalized, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, cv2.BORDER_CONSTANT, 0
    )

    # Spatial Filtering
    GF = GuidedFilter(leaf_image, 8, 0.4 ** 2)
    mask_gri_normalized = GF.filter(mask_gri_normalized)
    mask_gri_normalized[mask_gri_normalized < 0] = 0
    mask_gri_normalized[mask_gri_normalized >= 1.0] = 1.0
    mask_gri_normalized[lsh_mask == 0] = 0
    mask_gri_colorized = cv2.applyColorMap((mask_gri_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask_gri_colorized[lsh_where[0], lsh_where[1], :] = (0, 0, 0)
    # show(mask_gri_colorized)

    cv2.imwrite(str(Path(input_name).stem + "_output.png"), cv2.hconcat((leaf_image, cv2.addWeighted(leaf_image, 0.7, mask_gri_colorized, 0.3, 0))))
    cv2.waitKey(10)


if __name__ == "__main__":
    main(sys.argv)