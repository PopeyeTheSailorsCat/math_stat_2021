import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt


def function(x):
    return 2 + 2 * x


def noisy_function(x):
    y = [function(i) + stats.norm.rvs(0, 1) for i in x]
    return y


def distance(y_model, y_regr):
    dist_y = sum([(y_model[i] - y_regr[i]) ** 2 for i in range(len(y_model))])
    return dist_y


def mnm_parameters(x, y):
    beta_0, beta_1 = mnk_parameters(x, y)
    result = opt.minimize(min_module_method, [beta_0, beta_1], args=(x, y), method='SLSQP')
    coefs = result.x
    alpha_0, alpha_1 = coefs[0], coefs[1]
    return alpha_0, alpha_1


def mnk_parameters(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def min_module_method(parameters, x, y):
    alpha_0, alpha_1 = parameters
    sum = 0
    for i in range(len(x)):
        sum += abs(y[i] - alpha_0 - alpha_1 * x[i])
    return sum


def MNK(x, y):
    beta_0, beta_1 = mnk_parameters(x, y)
    print("MNK")
    print('b_0 =  ', str(beta_0), 'b_1 = ', str(beta_1))
    y_new = [beta_0 + beta_1 * elem for elem in x]
    return y_new


def MNM(x, y):
    alpha_0, alpha_1 = mnm_parameters(x, y)
    print("MNM")
    print('a_0= ' + str(alpha_0), 'a_1 = ' + str(alpha_1))
    y_new = [alpha_0 + alpha_1 * x_ for x_ in x]
    return y_new


def run_and_plot_linear_regression(text, x, y):
    y_mnk = MNK(x, y)
    y_dist_mnk = distance(y, y_mnk)
    print('mnk dist', y_dist_mnk)
    plt.plot(x, y_mnk, label="МНК", color='blue')

    y_mnm = MNM(x, y)
    y_dist_mnm = distance(y, y_mnm)
    print('mnm dist', y_dist_mnm)

    plt.plot(x, y_mnm, label="МНМ", color='green')
    plt.plot(x, function(x), color='red', label='Модель')
    plt.scatter(x, y, c='black', label='Выборка')

    plt.xlim([-1.8, 2])
    plt.grid()
    plt.legend()
    plt.savefig(text + '.png', format='png')
    plt.show()


x = np.arange(-1.8, 2, 0.2)
y = noisy_function(x)
run_and_plot_linear_regression('NoPerturb', x, y)

y[0] += 10
y[-1] -= 10
run_and_plot_linear_regression('Perturb', x, y)
