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
import toml

SCRIPT_DIR = Path(__file__).parent.resolve()
CONF_PATH = SCRIPT_DIR / "conf/conf.toml"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR_RESULT = f"{OUTPUT_DIR}/result/"
OUTPUT_DIR_EXGREEN = f"{OUTPUT_DIR}/ex_green/"
OUTPUT_DIR_VALUES = f"{OUTPUT_DIR}/values/"
OUTPUT_DIR_CONCAT = f"{OUTPUT_DIR}/concat/"
OUTPUT_DIR_EVE = f"{OUTPUT_DIR}/eve/"

with open(str(CONF_PATH), "r") as f:
        config = toml.load(f)
print("-------------------------===========")

RGB_COLORMAP_3 = config["rgb_colormap_3"]
RGB_COLORMAP_5 = config["rgb_colormap_5"]
RGB_COLORMAP_10 = config["rgb_colormap_10"]
RGB_COLORMAP_10_2 = config["rgb_colormap_10_2"]

RGB_AREA_3 = config["rgb_area_3"]
RGB_AREA_5 = config["rgb_area_5"]
RGB_AREA_10 = config["rgb_area_10"]
RGB_AREA_10_2 = config["rgb_area_10_2"]

LSH_LOWER = config["lsh_lower"]
LSH_UPPER = config["lsh_upper"]
GRID_SIZE = config["grid_size"]


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

# def get_green_area(leaf_image_bgr,lsh_mask):
#     leaf_image_float = leaf_image_bgr / 255
#     leaf_image_float += 10e-8
#     lsh_where = np.where(lsh_mask == 0)
#     leaf_image_float[lsh_where[0], lsh_where[1], :] = (1.0, 0.1, 1.0)
#     is_green_dominant = (np.max(leaf_image_float[:, ..., :] / leaf_image_float[:, :, 1][:, :, np.newaxis], axis=2) > 1.0) & (
#         np.max(leaf_image_float, axis=2) > 0.5
#     )
#     lsh_mask = np.zeros_like(is_green_dominant, np.uint8)
#     lsh_mask[is_green_dominant] = 255
#     return lsh_mask

def get_grid_view(lsh_mask, num_divided_width=9):
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

# colorconverter
def convert_color(output_img, condition_img, low_condition, high_condition, value):
    img_out = output_img.copy() 
    if low_condition == 0:
        img_out[(low_condition <= condition_img) & (condition_img <= high_condition)] = value
    else:
        img_out[(low_condition < condition_img) & (condition_img <= high_condition)] = value
    return img_out

def make_color_map(shape, green_average, rgb_colormap, value_area): 
    green_average_out = np.zeros(shape)
    rgb_colormap = rgb_colormap 
    value_area = value_area 

    for i in range(len(rgb_colormap)):
        print(i,value_area[i], value_area[i+1])
        green_average_out = convert_color(green_average_out, green_average, value_area[i], value_area[i+1], rgb_colormap[i]) 
    return green_average_out

def make_color_map_5(shape, green_average): 
    green_average_out = np.zeros(shape)
    value_area = [0,0.7,0.85,0.9,0.99,1.0]
    for i in range(len(RGB_COLORMAP_5)):
        print(i,value_area[i], value_area[i+1])
        green_average_out = convert_color(green_average_out, green_average, value_area[i], value_area[i+1], RGB_COLORMAP_5[i])
    return green_average_out

def make_color_map_10(shape, green_average): 
    green_average_out = np.zeros(shape)
    value_area = [0,0.2,0.4,0.5,0.7,0.8,0.85,0.9,0.95,0.99,1.0]
    for i in range(len(RGB_COLORMAP_10)):
        print(i,value_area[i], value_area[i+1])
        green_average_out = convert_color(green_average_out, green_average, value_area[i], value_area[i+1], RGB_COLORMAP_10[i])
    return green_average_out

def alpha_blend(img1, img2, alpha=0.5):
    overlay_img = (alpha * img1 + (1-alpha) * img2).astype(np.uint8)
    return overlay_img

def alpha_blend_2(img1, img2, green_average, alpha=0.5):
    change_pixcel = np.where(img1 < 0.8)
    overlay_img = (alpha * img1 + (1-alpha) * img2).astype(np.uint8)
    overlay_img[green_average <= 0.8] = img1[green_average <= 0.8]
    return overlay_img

# def imwrite(input_name, overlay_img_1, overlay_img_2, overlay_img_3 ,lsh_mask, leaf_image_bgr):
#     overlay_img_1 = cv2.cvtColor(overlay_img_1, cv2.COLOR_RGB2BGR)
#     overlay_img_2 = cv2.cvtColor(overlay_img_2, cv2.COLOR_RGB2BGR)
#     overlay_img_3 = cv2.cvtColor(overlay_img_3, cv2.COLOR_RGB2BGR)
# 
#     lsh_mask = np.concatenate([lsh_mask[:,:,None],lsh_mask[:,:,None],lsh_mask[:,:,None]],2)
#     im_h = cv2.hconcat([leaf_image_bgr, lsh_mask, overlay_img_1, overlay_img_2, overlay_img_3])
#     cv2.imwrite(str(OUTPUT_DIR_EVE + Path(input_name).stem + "_output.png"), im_h)

def imwrite(input_name, overlay_img_1, overlay_img_2):
    overlay_img_1 = cv2.cvtColor(overlay_img_1, cv2.COLOR_RGB2BGR)
    overlay_img_2 = cv2.cvtColor(overlay_img_2, cv2.COLOR_RGB2BGR)
    im_h = cv2.hconcat([overlay_img_1, overlay_img_2])
    cv2.imwrite(str(OUTPUT_DIR_EVE + Path(input_name).stem + "_output.png"), im_h)

def main(args):
    input_name = args[1] if len(args) > 1 else "data/leaf.jpg"
    leaf_image_bgr = read_image(input_name)
    leaf_image_rgb = cv2.cvtColor(leaf_image_bgr, cv2.COLOR_BGR2RGB)
    img_lch = rgb2lch(leaf_image_rgb)
    img_lsh = calculate_perception(img_lch) 
    lsh_mask = extract_blank_area(img_lsh, leaf_image_bgr)
    #mask_green_dominant = get_green_area(leaf_image_bgr,lsh_mask)

    overgrow_average_3, green_average_3, green_average_values_3 = \
            get_grid_view(lsh_mask, num_divided_width=GRID_SIZE)
    green_average_rgb = make_color_map(leaf_image_rgb.shape, green_average_3, RGB_COLORMAP_3, RGB_AREA_3)
    overlay_img_3 = alpha_blend(leaf_image_rgb, green_average_rgb)

    overgrow_average_5, green_average_5, green_average_values_5 = \
            get_grid_view(lsh_mask, num_divided_width=GRID_SIZE)
    green_average_rgb = make_color_map(leaf_image_rgb.shape, green_average_5, RGB_COLORMAP_5, RGB_AREA_5)
    overlay_img_5 = alpha_blend(leaf_image_rgb, green_average_rgb)

    overgrow_average_10, green_average_10, green_average_values_10 = \
            get_grid_view(lsh_mask, num_divided_width=GRID_SIZE)
    green_average_rgb = make_color_map(leaf_image_rgb.shape, green_average_10, RGB_COLORMAP_10, RGB_AREA_10)
    overlay_img_10 = alpha_blend(leaf_image_rgb, green_average_rgb)

    overgrow_average_10_1, green_average_10_1, green_average_values_10_1 = \
            get_grid_view(lsh_mask, num_divided_width=GRID_SIZE)
    green_average_rgb = make_color_map(leaf_image_rgb.shape, green_average_10_1, RGB_COLORMAP_10_2, RGB_AREA_10)
    overlay_img_10_1 = alpha_blend_2(leaf_image_rgb, green_average_rgb, green_average_10_1)

    overgrow_average_10_2, green_average_10_2, green_average_values_10_2 = \
            get_grid_view(lsh_mask, num_divided_width=GRID_SIZE)
    green_average_rgb = make_color_map(leaf_image_rgb.shape, green_average_10_2, RGB_COLORMAP_10, RGB_AREA_10_2)
    overlay_img_10_2 = alpha_blend_2(leaf_image_rgb, green_average_rgb, green_average_10_2)

    imwrite(input_name, overlay_img_10_1, overlay_img_10_2)

if __name__ == "__main__":
    main(sys.argv)
