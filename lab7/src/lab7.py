import numpy as np
from scipy.stats import laplace, uniform
import scipy.stats as stats
import math as math

selection_size = [20, 100]
ALPHA = 0.05
CONST_P = 1 - ALPHA


def calc_k_from_size(size):
    return math.ceil(1.72 * (size) ** (1 / 3))


def calculate(distribution, k):
    mu = np.mean(distribution)
    sigma = np.std(distribution)

    print('mu=' + str(np.around(mu, decimals=2)))
    print('sigma=' + str(np.around(sigma, decimals=2)))
    limits = np.linspace(-1.1, 1.1, num=k - 1)
    xi_2 = stats.chi2.ppf(CONST_P, k - 1)
    print('xi2=' + str(xi_2))
    return limits


def calculate_n_and_p(distribution, limits, size):
    P_list = np.array([])
    N_list = np.array([])
    for i in range(-1, len(limits)):
        if i != -1:
            prev_cdf_val = stats.norm.cdf(limits[i])
        else:
            prev_cdf_val = 0
            P_list = np.append(N_list, len(distribution[distribution <= limits[0]]))
        if i != len(limits) - 1:
            cur_cdf_val = stats.norm.cdf(limits[i + 1])
        else:
            cur_cdf_val = 1
            N_list = np.append(N_list, len(distribution[distribution >= limits[-1]]))
        if i != -1 and i != len(limits) - 1:
            N_list = np.append(N_list, len(distribution[(distribution <= limits[i + 1]) & (distribution >= limits[i])]))
        P_list = np.append(P_list, cur_cdf_val - prev_cdf_val)
    result = np.divide(np.multiply((N_list - size * P_list), (N_list - size * P_list)), P_list * size)
    return N_list, P_list, result


def run_distr_task(k, distr):
    limits = calculate(distr, k)
    n_list, p_list, result = calculate_n_and_p(distr, limits, len(distr))
    # output
    for i in range(0, len(n_list)):
        if i == 0:
            boarders = ['-inf', np.around(limits[0], decimals=2)]
        elif i == len(n_list) - 1:
            boarders = [np.around(limits[-1], decimals=2), 'inf']
        else:
            boarders = [np.around(limits[i - 1], decimals=2), np.around(limits[i], decimals=2)]

            print(i + 1, " ", boarders, " ", n_list[i], " ", np.around(p_list[i], decimals=4))
            print(np.around(p_list[i] * len(distr), decimals=2), " ",
                  np.around(n_list[i] - len(distr) * p_list[i], decimals=2))
            print(np.around(result[i], decimals=2))


k = calc_k_from_size(selection_size[1])
distr = np.random.normal(0, 1, size=selection_size[1])
run_distr_task(k, distr)

k = calc_k_from_size(selection_size[0])
distr = laplace.rvs(size=selection_size[0], scale=1 / math.sqrt(2), loc=0)
run_distr_task(k, distr)

k = calc_k_from_size(selection_size[0])
distr = uniform.rvs(size=selection_size[0], loc=-math.sqrt(3), scale=2 * math.sqrt(3))
run_distr_task(k, distr)
