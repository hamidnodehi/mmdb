import os
import sys
import math

import numpy as np
import cv2 as cv


def get_script_directory():
    return os.path.dirname(__file__)


DEFAULT_IMAGE_PATH: str = os.path.join(get_script_directory(), "images\\barbara.jpg")
DEFAULT_ORDER: int = 9
DEFAULT_SIGMA: float = 1.4


def display_and_wait_for_finish(image: np.ndarray, window_name: str):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow(window_name, image)
    while (cv.waitKey(10) & 0xFF) != 27 and cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) == 1:
        pass
    if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) == 1:
        cv.destroyWindow(window_name)


def display_kernel(kernel: np.ndarray, decimal_precision: int = 0, space: int = 5):
    if decimal_precision < 0 or space < 1:
        raise ValueError()
    format_string = "{{:{}.{}f}}".format(space, decimal_precision)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            print(format_string.format(kernel[i, j]), end="")
        print()


def laplace_of_gaussian(x: int, y: int, sigma: float) -> float:
    d = x ** 2 + y ** 2
    return (-1 / (math.pi * sigma ** 4)) * (1 - (d / (2 * sigma ** 2))) * math.exp(-d / (2 * sigma ** 2))


def create_log_kernel(order: int, sigma: float, scale: float) -> np.ndarray:
    if order < 1 or (order % 2) == 0:
        raise ValueError("invalid order")
    if sigma <= 0.:
        raise ValueError("invalid sigma")
    kernel = np.zeros((order, order), dtype=np.float32)
    for i in range(order):
        for j in range(order):
            kernel[i, j] = laplace_of_gaussian(j - order // 2, i - order // 2, sigma)
    return kernel * scale


def run():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    order = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_ORDER
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_SIGMA
    kernel = create_log_kernel(order, sigma, 1.)
    display_kernel(kernel, 5, 10)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image = np.array(image, dtype=np.float32)
    filtered_image = cv.filter2D(image, -1, kernel)
    filtered_min = filtered_image.min()
    filtered_max = filtered_image.max()
    filtered_image = (filtered_image - filtered_min) * (255. / (filtered_max - filtered_min))
    filtered_image = np.array(filtered_image, dtype=np.uint8)
    filtered_image = cv.convertScaleAbs(filtered_image, alpha=3., beta=-300.)
    display_and_wait_for_finish(filtered_image, "Result")


if __name__ == "__main__":
    run()
