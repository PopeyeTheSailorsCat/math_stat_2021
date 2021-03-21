from scipy.stats import laplace, uniform, norm, cauchy, poisson
import math as m
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from statsmodels.distributions.empirical_distribution import ECDF

rvs = 1
pdf = 2
cdf = 3
selection_size = [20, 60, 100]  # Размеры выборок


def laplace_distribution(select_size, scale=1 / m.sqrt(2), loc=0, asked=rvs, x=0):
    if asked == rvs:
        return laplace.rvs(size=select_size, scale=scale, loc=loc)
    elif asked == pdf:
        return laplace.pdf(x, loc=loc, scale=scale)
    elif asked == cdf:
        return laplace.cdf(x, loc=loc, scale=scale)
    return


def uniform_distribution(select_size, loc=-m.sqrt(3), scale=2 * m.sqrt(3), asked=rvs, x=0):
    if asked == rvs:
        return uniform.rvs(size=select_size, loc=loc, scale=scale)
    elif asked == pdf:
        return uniform.pdf(x, loc=loc, scale=scale)
    elif asked == cdf:
        return uniform.cdf(x, loc=loc, scale=scale)


def norm_distribution(select_size, loc=0, dist=1, asked=rvs, x=0):
    if asked == rvs:
        return norm.rvs(size=select_size)
    elif asked == pdf:
        return norm.pdf(x)
    elif asked == cdf:
        return norm.cdf(x)


def cauchy_distribution(select_size, asked=rvs, x=0):
    if asked == rvs:
        return cauchy.rvs(size=select_size)
    elif asked == pdf:
        return cauchy.pdf(x)
    elif asked == cdf:
        return cauchy.cdf(x)


def poisson_distribution(select_size, power=10, asked=rvs, x=0):
    if asked == rvs:
        return poisson.rvs(power, size=select_size)
    elif asked == pdf:
        return poisson(power).pmf(x)
    elif asked == cdf:
        return poisson.cdf(x, mu=power)


def Empirical():
    func = [[laplace_distribution, "LAPLACE"], [uniform_distribution, "UNIFORM"], [norm_distribution, "GAUSS"],
            [cauchy_distribution, "CAUCHY"], [poisson_distribution, "POISSON"]]
    sb.set_style('whitegrid')
    left = -4
    right = 4
    p_left = 4
    p_right = 16
    for f in func:
        funct, name = f
        figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
        count = 0
        for size in selection_size:
            data = funct(size)
            data.sort()
            if name == "POISSON":
                x = np.linspace(p_left, p_right, 10000)
            else:
                x = np.linspace(left, right, 10000)
            y = funct(size, asked=cdf, x=x)
            ecdf = ECDF(data)
            axs[count].plot(x, y, color='blue', label='cdf')
            axs[count].plot(x, ecdf(x), color='red', label='ecdf')
            axs[count].legend(loc='lower right')
            axs[count].set(xlabel='x', ylabel='F(x)')
            axs[count].set_title(name + ' n = ' + str(size))
            count += 1
        figures.savefig('img/' + name + ".png")
    return


def Nucleus():
    func = [[laplace_distribution, "LAPLACE"], [uniform_distribution, "UNIFORM"], [norm_distribution, "GAUSS"],
            [cauchy_distribution, "CAUCHY"], [poisson_distribution, "POISSON"]]
    sb.set_style('whitegrid')
    h_koefs = [0.5, 1, 2]
    left = -4
    right = 4
    p_left = 4
    p_right = 16
    for f in func:
        funct, name = f

        for size in selection_size:
            figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
            data = funct(size)
            data.sort()
            if name == "POISSON":
                x = np.linspace(p_left, p_right, p_right-p_left+1)
                begin = p_left
                end = p_right
            else:
                x = np.linspace(left, right, 10000)
                begin = left
                end = right
            count = 0
            for h_koef in h_koefs:
                y = funct(size, asked=pdf, x=x)
                axs[count].plot(x, y, color='red', label='pdf')

                sb.kdeplot(data=data, bw_method='scott', bw_adjust=h_koef, ax=axs[count],
                           fill=True, common_norm=False, palette="red", alpha=.5, linewidth=0.5, label='kde',
                           color="red")
                axs[count].legend(loc='upper right')
                axs[count].set(xlabel='x', ylabel='f(x)')
                axs[count].set_xlim([begin, end])
                axs[count].set_title('h = ' + str(h_koef))
                count += 1
            figures.suptitle(name + '_KDE n = ' + str(size))
            figures.savefig('kde/' + name + '_KDE_' + str(size) + ".png")
            # figures.savefig('kde/' + name + ".png")
    return


# Empirical()
Nucleus()
