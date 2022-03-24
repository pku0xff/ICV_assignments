import numpy as np
from utils import read_img, draw_corner
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y


def corner_response_function(I_xx, I_yy, I_xy, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            I_xx: array(float)
            I_yy: array(float) 
            I_xy: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: list
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # for detials of corner_response_function, please refer to the slides.
    h, w = I_xx.shape

    # It is aborted.
    def window_sum_with_for(Matrix):
        S = np.zeros((h - window_size, w - window_size))
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                S[i, j] = Matrix[i:i + window_size, j:j + window_size].sum()
        return S

    def window_sum(Matrix):
        kernel = np.ones(window_size)
        S = np.zeros((h + window_size, w + window_size))
        S[:h, :w] = Matrix
        S = np.convolve(S.reshape(-1), kernel, 'same')
        S = np.roll(S, -(window_size // 2)).reshape((h + window_size, w + window_size))

        S = S.T
        S = np.convolve(S.reshape(-1), kernel, 'same')
        S = np.roll(S, -(window_size // 2)).reshape(w + window_size, h + window_size).T

        S = S[:h + 1 - window_size, :w + 1 - window_size]
        return S

    S_xx = window_sum(I_xx)
    S_yy = window_sum(I_yy)
    S_xy = window_sum(I_xy)

    Det = S_xx * S_yy - S_xy ** 2
    Trace = S_xx + S_yy
    Theta = Det - alpha * Trace ** 2
    corner_x, corner_y = np.where(Theta > threshold)
    theta_value = Theta[corner_x, corner_y]
    corner_list = list(zip(corner_x + window_size // 2, corner_y + window_size // 2, theta_value))

    return corner_list  # the corners in corne_list: a tuple of (index of rows, index of cols, theta)


if __name__ == "__main__":

    # Load the input images
    input_img = read_img("hand_writting.png") / 255.

    I_x = Sobel_filter_x(input_img)
    I_y = Sobel_filter_y(input_img)

    I_xx = I_x ** 2
    I_yy = I_y ** 2
    I_xy = I_x * I_y

    I_xx = Gaussian_filter(I_xx)
    I_yy = Gaussian_filter(I_yy)
    I_xy = Gaussian_filter(I_xy)

    # you can adjust the parameters to fit your own implementation
    window_size = 5
    alpha = 0.04
    threshold = 10.

    corner_list = corner_response_function(I_xx, I_yy, I_xy, window_size, alpha, threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key=lambda x: x[2], reverse=True)
    NML_selected = []
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted:
        for j in NML_selected:
            if (abs(i[0] - j[0] <= dis) and abs(i[1] - j[1]) <= dis):
                break
        else:
            NML_selected.append(i[:-1])

    # save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
