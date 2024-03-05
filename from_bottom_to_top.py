import cv2
import numpy as np
from pathlib import Path
import click
from scripts.utils import create_output_dir, read_image
from scripts.leaf_density_index import (
    rgb2lch,
    extract_bright_area,
    extract_green_area,
    enhance_perception,
)
from scripts.config import LDIConfig


SCRIPT_DIR = Path(__file__).parent.resolve()


def calculate_pseudo_leaf_density_rate(image_with_border, leaf_area_mask, crop_ratio_h: int, border_color):
    """
    Calculates the pseudo leaf density rate within a specified crop area of the image and draws the area on the image.

    Args:
        image_with_border (numpy.ndarray): The original image where the crop area will be visualized with a border.
        leaf_area_mask (numpy.ndarray): The leaf area mask to calculate the density rate.
        crop_ratio_h (int): The ratio to determine the height of the crop area based on the image height.
        border_color (List[int]): The color of the border to draw the crop area.

    Returns:
        Tuple[float, numpy.ndarray, numpy.ndarray]:
            - The calculated pseudo leaf density rate within the crop area.
            - The original image with the crop area visualized by a border.
            - The pseudo leaf density mask with the crop area visualized by a border.
    """
    # Determine the dimensions of the image
    height, width, _ = leaf_area_mask.shape

    # Calculate the middle point of the image
    mid_point = (width // 2, height // 2)

    # Calculate the start and end points of the crop area based on the crop ratio
    start_height = mid_point[1] - int(height * crop_ratio_h / 2)
    end_height = mid_point[1] + int(height * crop_ratio_h / 2)

    # Crop the mask image to calculate the density rate
    cropped_mask = leaf_area_mask[start_height:end_height]

    # Calculate the rate of pseudo leaf density (ratio of white pixels in the cropped area)
    density_rate = np.sum(cropped_mask > 0) / cropped_mask.size

    # Draw the crop area on the original image and the mask image
    image_with_crop_border = cv2.rectangle(
        image_with_border, (0, start_height), (width, end_height), color=border_color, thickness=3
    )
    mask_with_crop_border = cv2.rectangle(
        leaf_area_mask, (0, start_height), (width, end_height), color=border_color, thickness=3
    )

    return density_rate, image_with_crop_border, mask_with_crop_border


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--conf-path",
    "conf_path",
    default=SCRIPT_DIR / "conf/conf_bottom.toml",
    type=click.Path(exists=True),
    help="設定ファイルのパス (TOML形式)",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=SCRIPT_DIR / "output",
    type=click.Path(),
    help="出力画像を保存するディレクトリのパス",
)
@click.option(
    "--output-csv-name",
    "output_csv_name",
    default="pseudo_leaf_density_rate_bottom.csv",
    type=click.Path(),
    help="出力csvの名前",
)
def main(input_path, conf_path, output_dir, output_csv_name):
    # 設定を読み込む
    config: LDIConfig = LDIConfig.from_toml(conf_path)

    # 出力ディレクトリを作成
    output_dir_pathlib = create_output_dir(output_dir)

    # 画像を読み込む
    input_pathlib = Path(input_path)
    leaf_image_bgr = read_image(input_path)
    leaf_image_rgb = cv2.cvtColor(leaf_image_bgr, cv2.COLOR_BGR2RGB)
    leaf_image_lsh = enhance_perception(rgb2lch(leaf_image_rgb))

    # 疑似葉密度マスクを作成する
    bright_area_mask = extract_bright_area(leaf_image_lsh, config.lsh_lower, config.lsh_upper)
    green_area_mask = extract_green_area(leaf_image_bgr, config.hsv_lower, config.hsv_upper)
    leaf_area_mask = 255 - (bright_area_mask & np.bitwise_not(green_area_mask))
    leaf_area_mask_3ch = cv2.cvtColor(leaf_area_mask, cv2.COLOR_GRAY2BGR)

    # 画像を分割して擬似葉密度を計算する
    border_draw_img = leaf_image_bgr.copy()
    rate_list = []
    for i in range(len(config.horizontal_crop_ratio_list)):
        density_rate, border_draw_img, leaf_area_mask_3ch = calculate_pseudo_leaf_density_rate(
            border_draw_img, leaf_area_mask_3ch, config.horizontal_crop_ratio_list[i], config.border_color[i]
        )
        rate_list.append(density_rate)

        # RoI: Region of Interest
        print(
            f"RoI height / Image height: {config.horizontal_crop_ratio_list[i]}, pseudo leaf density rate: {density_rate}"
        )

    # 結果をcsvに書き込む
    output_csv_path = str(output_dir_pathlib / output_csv_name)
    with open(output_csv_path, "a", encoding="UTF-8") as f:
        f.write(f"{output_dir},{input_pathlib.name},{rate_list[0]},{rate_list[1]},{rate_list[2]},{rate_list[3]}\n")
    print(f"Result CSV was saved at {output_csv_path}")

    # 結果画像を出力する
    img_h = cv2.hconcat([border_draw_img, leaf_area_mask_3ch])
    output_image_path = str(output_dir_pathlib / f"{input_pathlib.stem}_with_border.jpg")
    cv2.imwrite(output_image_path, img_h, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"Result image was saved at {output_image_path}")


if __name__ == "__main__":
    main()
