from scipy.stats import laplace, uniform, norminvgauss, cauchy, poisson
import matplotlib.pyplot as plt
import math
import numpy as np

# global var
selection_size = [10, 50, 1000]  # Размеры выборок
HIST_TYPE = 'stepfilled'  # Вид нашей гистограммы
LINE = 'k-.'
hist_visibility = 0.5
line_width = 1

TITLE = "Selection size: "  # Заголовок в каждом графике
Y_LABEL = "Probability density"


def Hist_Laplace():
    laplace_scale = 1 / math.sqrt(2)  # Параметры
    laplace_loc = 0
    laplace_label = "Laplace distribution"
    laplace_color = "blue"
    for size in selection_size:
        fig, ax = plt.subplots(1, 1)
        pdf = laplace(scale=laplace_scale, loc=laplace_loc)
        random_values = laplace.rvs(size=size, scale=laplace_scale, loc=laplace_loc)
        ax.hist(random_values, density=True, histtype=HIST_TYPE, alpha=hist_visibility, color=laplace_color)
        Create_plot(ax, laplace_label, pdf, size)


def Hist_Uniform():
    uniform_scale = 2 * math.sqrt(3)  # Параметры
    uniform_loc = -math.sqrt(3)
    uniform_label = "Uniform distribution"
    uniform_color = "red"
    for size in selection_size:
        fig, ax = plt.subplots(1, 1)
        pdf = uniform(scale=uniform_scale, loc=uniform_loc)
        random_values = uniform.rvs(size=size, scale=uniform_scale, loc=uniform_loc)
        ax.hist(random_values, density=True, histtype=HIST_TYPE, alpha=hist_visibility, color=uniform_color)
        Create_plot(ax, uniform_label, pdf, size)


def Hist_Normal():
    normal_scale = 1  # Параметры
    normal_loc = 0
    normal_label = "Normal distribution"
    normal_color = "green"
    for size in selection_size:
        fig, ax = plt.subplots(1, 1)
        pdf = norminvgauss(normal_scale, normal_loc)
        random_values = norminvgauss.rvs(normal_scale, normal_loc, size=size)
        ax.hist(random_values, density=True, histtype=HIST_TYPE, alpha=hist_visibility, color=normal_color)
        Create_plot(ax, normal_label, pdf, size)


def Hist_Cauchy():
    cauchy_scale = 1  # Параметры
    cauchy_loc = 0
    cauchy_label = "Cauchy distribution"
    cauchy_color = "grey"
    for size in selection_size:
        fig, ax = plt.subplots(1, 1)
        pdf = cauchy()
        random_values = cauchy.rvs(size=size)
        ax.hist(random_values, density=True, histtype=HIST_TYPE, alpha=hist_visibility, color=cauchy_color)
        Create_plot(ax, cauchy_label, pdf, size)


def Hist_Poisson():
    poisson_param = 10  # Параметры
    poisson_label = "Poisson distribution"
    poisson_color = "violet"
    for size in selection_size:
        fig, ax = plt.subplots(1, 1)
        pdf = poisson(poisson_param)
        random_values = poisson.rvs(poisson_param, size=size)
        ax.hist(random_values, density=True, histtype=HIST_TYPE, alpha=hist_visibility, color=poisson_color)
        # Create_plot(ax, poisson_label, pdf, size) Пуассон не такой
        x = np.arange(poisson.ppf(0.01, 10), poisson.ppf(0.99, 10))
        ax.plot(x, pdf.pmf(x), LINE, lw=line_width)
        ax.set_xlabel(poisson_label)
        ax.set_ylabel(Y_LABEL)
        ax.set_title(TITLE + str(size))
        plt.grid()
        plt.show()


def Create_plot(ax, x_label, pdf, size):
    x = np.linspace(pdf.ppf(0.01), pdf.ppf(0.99), 100)
    ax.plot(x, pdf.pdf(x), LINE, lw=line_width)
    ax.set_xlabel(x_label)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(TITLE + str(size))
    plt.grid()
    plt.show()


Hist_Laplace()
Hist_Uniform()
Hist_Normal()
Hist_Cauchy()
Hist_Poisson()
