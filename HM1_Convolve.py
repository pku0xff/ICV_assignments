import numpy as np
from utils import read_img, write_img


def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type == "zeroPadding":
        padding_img = np.zeros((img.shape[0] + padding_size * 2, img.shape[1] + padding_size * 2))
        padding_img[padding_size:img.shape[0] + padding_size, padding_size:img.shape[1] + padding_size] = img
        return padding_img
    elif type == "replicatePadding":
        up = np.repeat(np.expand_dims(img[0, :], axis=0), padding_size, axis=0)
        down = np.repeat(np.expand_dims(img[-1, :], axis=0), padding_size, axis=0)
        padding_img = np.vstack((up, img, down))
        left = np.repeat(np.expand_dims(padding_img[:, 0], axis=1), padding_size, axis=1)
        right = np.repeat(np.expand_dims(padding_img[:, -1], axis=1), padding_size, axis=1)
        padding_img = np.hstack((left, padding_img, right))
        return padding_img
    elif type == "reflectionPadding":
        # https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html
        if padding_size > min(img.shape[0], img.shape[1]):
            assert ValueError
        up = np.flip(img[-padding_size:, :].reshape((padding_size, img.shape[1])), axis=0)
        down = np.flip(img[:padding_size, :].reshape((padding_size, img.shape[1])), axis=0)
        left = np.flip(img[:, -padding_size:].reshape((img.shape[0], padding_size)), axis=1)
        right = np.flip(img[:, :padding_size].reshape((img.shape[0], padding_size)), axis=1)
        left_up = np.flip(np.flip(img[-padding_size:, -padding_size:], axis=0), axis=1)
        left_down = np.flip(np.flip(img[:padding_size, -padding_size:], axis=0), axis=1)
        right_up = np.flip(np.flip(img[-padding_size:, :padding_size], axis=0), axis=1)
        right_down = np.flip(np.flip(img[:padding_size, :padding_size], axis=0), axis=1)
        left = np.vstack((left_up, left, left_down))
        mid = np.vstack((up, img, down))
        right = np.vstack((right_up, right, right_down))
        padding_img = np.hstack((left, mid, right))
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # zero padding
    padding_img = padding(img, 1, "zeroPadding")  # the output size remains the same

    # build the Toeplitz matrix and compute convolution
    # Construct sub-matrix first, then concat them to form the Toeplitz matrix.
    # rows and cols are used to construct sub-matrix.
    # The size of a sub-matrix is 6*8. There are 6*8 sub-matrix.
    rows = np.array([[0, 0, 0],
                     [1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3],
                     [4, 4, 4],
                     [5, 5, 5]])
    cols = np.array([[0, 1, 2],
                     [1, 2, 3],
                     [2, 3, 4],
                     [3, 4, 5],
                     [4, 5, 6],
                     [5, 6, 7]])

    def construct_sub_matrix(i):
        sub_matrix = np.zeros((6, 8))
        sub_matrix[rows, cols] = np.tile(kernel[i], (6, 1))
        return sub_matrix

    mz = np.zeros((6, 8))
    m0, m1, m2 = construct_sub_matrix(0), construct_sub_matrix(1), construct_sub_matrix(2)

    toeplitz = np.block([[m0, m1, m2, mz, mz, mz, mz, mz],
                         [mz, m0, m1, m2, mz, mz, mz, mz],
                         [mz, mz, m0, m1, m2, mz, mz, mz],
                         [mz, mz, mz, m0, m1, m2, mz, mz],
                         [mz, mz, mz, mz, m0, m1, m2, mz],
                         [mz, mz, mz, mz, mz, m0, m1, m2]])

    output = toeplitz.dot(padding_img.flatten()).reshape(img.shape)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """

    # build the sliding-window convolution here
    # xufan: For every element in kernel, calculate them respectively, then sum up the 9 elements.
    #        The defect is that the size of kernel is limited. If for loop is permitted, the problem can be solved.
    x = np.arange(img.shape[0] - kernel.shape[0] + 1)
    y = np.arange(img.shape[1] - kernel.shape[1] + 1)
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.T, yv.T
    output = img[xv, yv] * kernel[0, 0] + img[xv, yv + 1] * kernel[0, 1] + img[xv, yv + 2] * kernel[0, 2] + \
             img[xv + 1, yv] * kernel[1, 0] + img[xv + 1, yv + 1] * kernel[1, 1] + img[xv + 1, yv + 2] * kernel[1, 2] + \
             img[xv + 2, yv] * kernel[2, 0] + img[xv + 2, yv + 1] * kernel[2, 1] + img[xv + 2, yv + 2] * kernel[2, 2]

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":
    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png") / 255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x * 255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y * 255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur * 255)
