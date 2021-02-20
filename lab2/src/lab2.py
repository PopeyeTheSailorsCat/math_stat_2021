from scipy.stats import laplace, uniform, norminvgauss, cauchy, poisson
import math as m
import numpy as np

# global var
selection_size = [10, 100, 1000]  # Размеры выборок
NUMBER_OF_EXPERIMENT = 1000


# general function:

def mean(selection):
    return np.mean(selection)


def median(selection):
    return np.median(selection)


def half_sum_extreme(selection, this_selection_size):
    return (selection[0] + selection[this_selection_size - 1]) / 2


def quantile_p(selection, NP):
    if NP.is_integer():
        return selection[int(NP)]
    else:
        return selection[int(NP) + 1]


def half_quantile_sum(selection, this_selection_size):
    quan_1 = quantile_p(selection, this_selection_size / 4)
    quan_2 = quantile_p(selection, 3 * this_selection_size / 4)
    return (quan_1 + quan_2) / 2


def trunc_mean(selection, this_selection_size):
    r = round(this_selection_size / 4)
    sum_ = 0
    for i in range(r + 1, this_selection_size - r + 1):
        sum_ += selection[i]
    return (1 / (this_selection_size - 2 * r)) * sum_


def laplace_distribution(select_size, scale=1 / m.sqrt(2), loc=0):
    return laplace.rvs(size=select_size, scale=scale, loc=loc)


def uniform_distribution(select_size, loc=-m.sqrt(3), scale=2 * m.sqrt(3)):
    return uniform.rvs(size=select_size, loc=loc, scale=scale)


def gauss_distribution(select_size, loc=0, dist=1):
    return norminvgauss.rvs(dist, loc, size=select_size)


def cauchy_distribution(select_size):
    return cauchy.rvs(size=select_size)


def poisson_distribution(select_size, power=10):
    return poisson.rvs(power, size=select_size)


def dispersion(selection):
    delta = np.std(selection)
    return delta * delta


def run_tests():
    func_name = ["LAPLACE", "UNIFORM", "GAUSS", "CAUCHY", "POISSON"]
    func = [[laplace_distribution, "LAPLACE"], [uniform_distribution, "UNIFORM"], [gauss_distribution, "GAUSS"],
            [cauchy_distribution, "CAUCHY"], [poisson_distribution, "POISSON"]]

    for f in func:
        print(f[1])
        for size in selection_size:
            mean_list, med_list, half_sum_list, half_quantile_list, trunc_mean_list = [], [], [], [], []
            lists = [mean_list, med_list, half_sum_list, half_quantile_list, trunc_mean_list]
            Exp = []
            Distr = []
            for i in range(NUMBER_OF_EXPERIMENT):
                # print(f[0](size))
                dist = f[0](size)
                mean_list.append(mean(dist))
                med_list.append(median(dist))
                half_sum_list.append(half_sum_extreme(dist, size))
                half_quantile_list.append(half_quantile_sum(dist, size))
                trunc_mean_list.append(trunc_mean(dist, size))
            for list in lists:
                Exp.append(round(mean(list), 7))
                Distr.append(round(dispersion(list), 7))
            print("E", Exp)
            print("D", Distr)


run_tests()
