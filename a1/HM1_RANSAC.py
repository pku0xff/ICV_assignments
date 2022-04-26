import math

import numpy as np
from utils import draw_save_plane_with_points

if __name__ == "__main__":
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")  # 130*3

    # RANSAC
    # we recommend you to formulate the palnace function as:  A*x+B*y+C*z+D=0    

    # 1-(1-w^k)^N = p
    # w=10/13, probability that a point is an inlier
    # k=3, number of points in a sample
    # N=?, number of samples
    # p>=99.9%, probability that we get a good example
    # more than 99.9% probability at least one hypothesis does not contain any outliers
    N = noise_points.shape[0]  # 130
    sample_time = math.ceil(math.log(0.001, (1 - (10 / 13) ** 3)))  # 12
    distance_threshold = 0.05

    # sample points group
    # Every group contains 3 points and there are sample_time groups in total.
    sample_idx = np.random.randint(0, N, size=(sample_time, 3))  # 12*3

    # estimate the plane with sampled points group
    sample_points = noise_points[sample_idx]  # 12*3*3


    def my_idx(xyz, p_idx):
        d_xyz = {'x': 0, 'y': 1, 'z': 2}
        d_idx = d_xyz[xyz]
        return sample_points[:, p_idx - 1, d_idx]


    A = (my_idx('y', 2) - my_idx('y', 1)) * (my_idx('z', 3) - my_idx('z', 1)) - \
        (my_idx('y', 3) - my_idx('y', 1)) * (my_idx('z', 2) - my_idx('z', 1))
    B = (my_idx('z', 2) - my_idx('z', 1)) * (my_idx('x', 3) - my_idx('x', 1)) - \
        (my_idx('z', 3) - my_idx('z', 1)) * (my_idx('x', 2) - my_idx('x', 1))
    C = (my_idx('x', 2) - my_idx('x', 1)) * (my_idx('y', 3) - my_idx('y', 1)) - \
        (my_idx('x', 3) - my_idx('x', 1)) * (my_idx('y', 2) - my_idx('y', 1))
    ABC = np.vstack((A, B, C)).T
    D = -A * my_idx('x', 1) - B * my_idx('y', 1) - C * my_idx('z', 1)

    # the distance between (x0, y0, z0) and Ax + By + C + D = 0 is
    # dist = abs(A*x0 + B*y0 + C*z0 + D) / sqrt(A**2 + B**2 + C**2)
    dist = np.abs(ABC @ noise_points.T + D[:, np.newaxis]) / np.sqrt(np.sum(ABC ** 2, axis=1)[:, np.newaxis] + 1e-20)
    # evaluate inliers (with point-to-plance distance < distance_threshold)
    count_inlier = np.where(dist < distance_threshold, 1, 0)
    # plane[max_idx] has the most inliers.
    max_idx = np.argmax(np.sum(count_inlier, -1))
    inlier_idx = np.where(count_inlier[max_idx] > 0)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    points = noise_points[inlier_idx]
    # We normalize D to -1.
    normal_vector = np.linalg.inv(points.T @ points) @ points.T @ np.ones((points.shape[0], 1))
    pf = list(normal_vector) + [-1]

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
