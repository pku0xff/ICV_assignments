import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.hypot(x_grad, y_grad)
    magnitude_grad = magnitude_grad / magnitude_grad.max()
    direction_grad = np.arctan2(x_grad, y_grad)  # I'm confused at the order of x_grad and y_grad...

    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """

    # r=q+g, p=q-g
    # perform bilinear interpolation to get the values at p and r
    # maximum if the value is larger
    def bilinear_interpolation(img, x, y):
        # DEBUGGING LOG: Don't use np.ceil() to get x2,y2.
        x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
        x2, y2 = x1 + 1, y1 + 1
        x1, x2 = np.clip(x1, 0, img.shape[0] - 1), np.clip(x2, 0, img.shape[0] - 1)
        y1, y2 = np.clip(y1, 0, img.shape[1] - 1), np.clip(y2, 0, img.shape[1] - 1)
        # element-wise multiplication
        return img[x1, y1] * (x2 - x) * (y2 - y) + img[x1, y2] * (x2 - x) * (y - y1) + \
               img[x2, y1] * (x - x1) * (y2 - y) + img[x2, y2] * (x - x1) * (y - y1)

    x = np.arange(grad_mag.shape[0])
    y = np.arange(grad_mag.shape[1])
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.T, yv.T
    direct1 = bilinear_interpolation(grad_mag, xv + np.cos(grad_dir), yv + np.sin(grad_dir))
    direct2 = bilinear_interpolation(grad_mag, xv - np.cos(grad_dir), yv - np.sin(grad_dir))
    NMS_output = np.where(np.logical_and(grad_mag > direct1, grad_mag > direct2), grad_mag, 0)

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """

    # you can adjust the parameters to fit your own implementation
    # Double threshold:
    # 1. Strong pixels are pixels that have an intensity so high that
    #    we are sure they contribute to the final edge.
    # 2. Weak pixels are pixels that have an intensity value that is
    #    not enough to be considered as strong ones, but yet not small
    #    enough to be considered as non-relevant for the edge detection.
    # 3. Other pixels are considered as non-relevant for the edge.
    #
    # Low threshold is used to identify the non-relevant pixels
    # (intensity lower than the low threshold)
    # High threshold is used to identify the strong pixels
    # (intensity higher than the high threshold)
    low_ratio = 0.07
    high_ratio = 0.20

    # remove the non-relevant pixels
    img = np.where(img > low_ratio, img, 0)
    # obtain the strong and weak pixels
    strong_pixels = np.where(img >= high_ratio, np.ones(img.shape), np.zeros(img.shape))
    weak_pixels = np.where(np.logical_and(img >= low_ratio, img < high_ratio), np.ones(img.shape), np.zeros(img.shape))
    output = strong_pixels

    def strengthen(i, j):
        l, r, d, u = max(i - 1, 0), min(i + 1, img.shape[0] - 1), max(j - 1, 0), min(j + 1, img.shape[1] - 1)
        return strong_pixels[l, d] or strong_pixels[l, j] or strong_pixels[l, u] or \
               strong_pixels[i, d] or strong_pixels[i, u] or \
               strong_pixels[r, d] or strong_pixels[r, j] or strong_pixels[r, u]

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if weak_pixels[i, j]:
                if strengthen(i, j):
                    output[i, j] = 1

    # TODO: Why is this algprithm differ from for loop?
    '''
    # Optimized:
    strong_x, strong_y = np.where(img >= high_ratio)
    weak_x, weak_y = np.where(np.logical_and(img >= low_ratio, img < high_ratio))
    up = np.clip(strong_x + 1, 0, img.shape[0] - 1)
    down = np.clip(strong_x - 1, 0, img.shape[0] - 1)
    left = np.clip(strong_y + 1, 0, img.shape[1] - 1)
    right = np.clip(strong_y - 1, 0, img.shape[1] - 1)
    output = np.zeros(img.shape)
    output[weak_x, weak_y] += 8
    output[up, left] += 1
    output[up, strong_y] += 1
    output[up, right] += 1
    output[strong_x, left] += 1
    output[strong_x, right] += 1
    output[down, left] += 1
    output[down, strong_y] += 1
    output[down, right] += 1
    output -= 8
    output = output.clip(0, 1)
    output[strong_x, strong_y] = 1
    '''

    return output


if __name__ == "__main__":
    # Load the input images
    input_img = read_img("Lenna.png") / 255

    # Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    # Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    # NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    # Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    # write_img("result/HM1_Canny_magnitude.png", magnitude_grad * 255)
    # write_img("result/HM1_Canny_direction.png", direction_grad * 255)
    # write_img("result/HM1_Canny_NMS.png", NMS_output * 255)
    write_img("result/HM1_Canny_result.png", output_img * 255)
