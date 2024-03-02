import cv2
import numpy as np
import click


@click.command()
@click.argument("input_path")
def main(input_path):
    img = cv2.imread(input_path)
    h, w, _ = img.shape

    if w > h:
        rotate_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(input_path, rotate_img)
        print(f"This image({input_path}) was rotated")
    else:
        print(f"This image's aspect({input_path}) is the shape that hight is longer than width.")


if __name__ == "__main__":
    main()
