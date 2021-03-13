from scipy.stats import laplace, uniform, norminvgauss, cauchy, poisson
import math as m
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

selection_size = [20, 100]  # Размеры выборок
NUMBER_OF_EXPERIMENT = 1000
LINE_20 = 'Доля выбросов для 20 элементов '
LINE_100 = 'Доля выбросов для 100 элементов '
SAVE_FORMAT = '.jpg'
SAVE_TO = 'img/'


def calc_moustache(distr):
    quan_1, quan_3 = np.quantile(distr, [0.25, 0.75])
    low_border = quan_1 - 3 / 2 * (quan_3 - quan_1)
    up_border = quan_3 + 3 / 2 * (quan_3 - quan_1)
    return low_border, up_border


def count_blowout(distr):
    low_border, up_border = calc_moustache(distr)
    return len([elem for elem in distr if elem > up_border or elem < low_border])


def render_boxplot(data, name):
    sb.set_theme(style="ticks")
    sb.boxplot(data=data, palette='rainbow', orient='h')
    sb.despine(offset=10)
    # plt.figure()
    plt.xlabel("x")
    plt.ylabel("n")
    plt.title(name)
    plt.savefig(SAVE_TO + str(name) + SAVE_FORMAT)
    plt.clf()
    return


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


def main():
    func = [[laplace_distribution, "LAPLACE"], [uniform_distribution, "UNIFORM"], [gauss_distribution, "GAUSS"],
            [cauchy_distribution, "CAUCHY"], [poisson_distribution, "POISSON"]]
    fun_pos = 0
    name_pos = 1
    for f in func:
        data, blowouts = [], []
        count = 0
        for size in selection_size:
            for i in range(NUMBER_OF_EXPERIMENT):
                distr = f[fun_pos](size)
                # print(distribution)
                # distribution.sort()
                count += count_blowout(distr)
            blowouts.append(count / (size * NUMBER_OF_EXPERIMENT))
            distr = f[fun_pos](size)
            # distribution.sort()
            data.append(distr)
        # print(data)
        render_boxplot(data, f[name_pos])
        print(f[name_pos])
        print(LINE_20 + str(round(blowouts[0], 3)))
        print(LINE_100 + str(round(blowouts[1], 3)))


main()
