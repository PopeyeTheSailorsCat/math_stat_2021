import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statistics
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

selection_size = [20, 60, 100]
cor_coefficients = [0, 0.5, 0.9]
repeat = 1000


def two_dim_norm(size, cor_coefficient):
    return stats.multivariate_normal.rvs([0, 0], [[1.0, cor_coefficient], [cor_coefficient, 1.0]], size=size)


def mix_two_dim_norm(size, cor_coefficient):
    first_coeff = 0.9
    second_coeff = 0.1
    return first_coeff * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + \
           second_coeff * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)


def quadrant_cor(x, y):
    x_med, y_med = np.median(x), np.median(y)
    counter = {1: 0, 2: 0, 3: 0, 4: 0}
    for x_i, y_i in zip(x, y):
        if x_i < x_med and y_i < y_med:
            counter[3] += 1
        elif x_i < x_med and y_i >= y_med:
            counter[2] += 1
        elif x_i >= x_med and y_i < y_med:
            counter[4] += 1
        else:
            counter[1] += 1
    return (counter[1] + counter[3] - counter[2] - counter[4]) / len(x)


def count_correlation_coefficient(sample, size, cor_coefficient):
    pearson = []
    quadrant = []
    spearman = []
    for i in range(repeat):
        this_sample = sample(size, cor_coefficient)
        x, y = this_sample[:, 0], this_sample[:, 1]
        pearson.append(stats.pearsonr(x, y)[0])
        spearman.append(stats.spearmanr(x, y)[0])
        quadrant.append(quadrant_cor(x, y))
    return pearson, spearman, quadrant


def create_pos_ellipse(x, y, ax, n_std=3.0):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rad_x = np.sqrt(1 + pearson)
    rad_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', edgecolor='black')
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def show_pos_ellipse(size):
    fig, ax = plt.subplots(1, 3)
    num = 0
    for coef in cor_coefficients:
        sample = two_dim_norm(size, coef)
        x, y = sample[:, 0], sample[:, 1]
        create_pos_ellipse(x, y, ax[num])
        ax[num].grid()
        ax[num].scatter(x, y, s=5)
        ax[num].set_title("n=" + str(size) + ',$ \\rho =$' + str(coef))
        num += 1
    plt.savefig("n_" + str(size) + ".png", format='png')
    plt.show()


def calc_res(selection):
    print("E(z)", np.round(np.median(selection), decimals=3))
    print("E(z^2)", np.round(np.median([selection[k] ** 2 for k in range(len(selection))]), decimals=3))
    print("D(z)", np.round(statistics.variance(selection), decimals=3))


for size in selection_size:
    print("size=", size)
    for cor in cor_coefficients:
        print("cor=", cor)
        pearson, spearman, quadrant = count_correlation_coefficient(two_dim_norm, size, cor)
        print("pear ")
        calc_res(pearson)
        print("spear ")
        calc_res(spearman)
        print('qand ')
        calc_res(quadrant)
        print("\n")
    print("\n\n")

    pearson, spearman, quadrant = count_correlation_coefficient(mix_two_dim_norm, size, 0)
    show_pos_ellipse(size)
    print("mixed")
    print("pear ")
    calc_res(pearson)
    print("spear ")
    calc_res(spearman)
    print('qand ')
    calc_res(quadrant)
    print("\n\n")
