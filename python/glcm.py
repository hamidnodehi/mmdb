import os
import sys

import numpy as np
import cv2 as cv


def get_script_directory():
    return os.path.dirname(__file__)


DEFAULT_IMAGE_PATH: str = os.path.join(get_script_directory(), "images\\barbara.jpg")
DEFAULT_DX: int = 20
DEFAULT_DY: int = 5
DEFAULT_ALPHA: float = 4.


def display_and_wait_for_finish(image: np.ndarray, window_name: str):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, image)
    while (cv.waitKey(10) & 0xFF) != 27 and cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) == 1:
        pass
    if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) == 1:
        cv.destroyWindow(window_name)


def generate_glcm_8_bit(image: np.ndarray, dx: int, dy: int, normalize: bool = False) -> np.ndarray:
    if len(image.shape) != 2 or image.dtype != np.uint8:
        raise ValueError("not a 8-bit grayscale image")
    glcm = np.zeros((256, 256), dtype=np.float64)
    w = image.shape[1]
    h = image.shape[0]
    for i in range(h):
        for j in range(w):
            u = i + dy
            v = j + dx
            if 0 <= u < h and 0 <= v < w:
                glcm[int(image[i, j]), int(image[u, v])] += 1
    if normalize:
        glcm = glcm / glcm.sum()
    return glcm


def glcm_8_bit_to_image(glcm: np.ndarray) -> np.ndarray:
    glcm = glcm / glcm.max() * 255.
    return np.array(glcm, dtype=np.uint8)


def run():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    dx = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DX
    dy = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DY
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_ALPHA
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    glcm = generate_glcm_8_bit(image, dx, dy)
    glcm = glcm_8_bit_to_image(glcm)
    glcm = cv.convertScaleAbs(glcm, alpha=alpha)
    display_and_wait_for_finish(glcm, "Result")


# def run_batch():
#     images_directory = os.path.join(get_script_directory(), "images\\glcm")
#     result_directory = os.path.join(images_directory, "result")
#     if not os.path.isdir(result_directory):
#         os.makedirs(result_directory)
#     for file_name in os.listdir(images_directory):
#         src_path = os.path.join(images_directory, file_name)
#         if not os.path.isfile(src_path):
#             continue
#         dst_path = os.path.join(result_directory, file_name)
#         image = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
#         glcm = generate_glcm_8_bit(image, DEFAULT_DX, DEFAULT_DY)
#         glcm = glcm_8_bit_to_image(glcm)
#         glcm = cv.convertScaleAbs(glcm, alpha=DEFAULT_ALPHA)
#         glcm = 255 - glcm
#         cv.imwrite(dst_path, glcm)


if __name__ == "__main__":
    run()
