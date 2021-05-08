import numpy as np
import matplotlib.pyplot as plt
import math


def read_signal_from_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            remove_dirst_str = line.replace("[", "")
            remove_next_str = remove_dirst_str.replace("]", "")
            data.append(remove_next_str.split(", "))
    data_float_format = []
    for item in data:
        data_float_format.append([float(x) for x in item])
    new_data = np.asarray(data_float_format)
    data = np.reshape(new_data, (new_data.shape[1] // 1024, 1024))
    return data[15]  # take data with index 15


def draw_signal_from_data(signal, title):
    plt.title(title)
    plt.plot(range(len(signal)), signal, 'blue')
    plt.grid()
    plt.savefig("signal_plot.png")
    plt.show()


def draw_hist(signal):
    counts, bins = np.histogram(signal)
    hist = plt.hist(bins[:-1], bins=bins, weights=counts, color='blue')
    plt.grid()
    plt.title("Signal hist")
    plt.savefig("hist.png")
    plt.show()


def draw_areas(signal_data, area_data, types, title):
    plt.title(title)
    plt.ylim([-0.1, 0.7])
    already_legend = set()
    for i in range(len(area_data)):
        if types[i] == "background":
            color_ = 'b'
        elif types[i] == "signal":
            color_ = 'r'
        elif types[i] == "transition":
            color_ = 'c'
        if types[i] not in already_legend:
            already_legend.add(types[i])

            plt.plot([num for num in range(area_data[i][0], area_data[i][1], 1)], signal_data[i], color=color_,
                     label=types[i])
        else:
            plt.plot([num for num in range(area_data[i][0], area_data[i][1], 1)], signal_data[i], color=color_)
    plt.grid()
    plt.legend()
    plt.savefig("separated.png")
    plt.show()


def get_intar_group_D(signal):
    result = 0.0
    for i in range(signal.shape[0]):
        mean = np.mean(signal[i])
        summ = 0.0
        for j in range(signal.shape[1]):
            summ += (signal[i][j] - mean) ** 2
        summ /= (signal.shape[0] - 1)
        result += summ

    return result / signal.shape[0]


def get_converted_data(signal, start, finish, types):
    signal_types = [0] * len(signal)
    zones = []
    zones_type = []

    for i in range(len(signal)):
        for j in range(len(types)):
            if (signal[i] >= start[j]) and (signal[i] <= finish[j]):
                signal_types[i] = types[j]

    currType = signal_types[0]
    start = 0
    for i in range(len(signal_types)):
        if currType != signal_types[i]:
            finish = i
            zones_type.append(currType)
            zones.append([start, finish])
            start = finish
            currType = signal_types[i]

    if currType != zones_type[len(zones_type) - 1]:
        zones_type.append(currType)
        zones.append([finish, len(signal) - 1])

    return zones, zones_type, get_signal_data(signal, zones)


def get_F(signal, k):
    newSizeY = int(signal.size / k)
    newSizeX = k
    print("k=" + str(k))
    split_data = np.reshape(signal, (newSizeX, newSizeY))
    inter_group = get_inter_group_D(split_data)
    print("Inter=" + str(inter_group))
    intra_group = get_intar_group_D(split_data)
    print("Intar=" + str(intra_group))
    print("F=" + str(inter_group / intra_group))
    return inter_group / intra_group


def get_inter_group_D(signal):
    summ = 0.0
    mean = np.empty(signal.shape[0])
    for i in range(len(signal)):
        mean[i] = np.mean(signal[i])
    meanMean = np.mean(mean)

    for i in range(len(mean)):
        summ += (mean[i] - meanMean) ** 2
    summ /= (signal.shape[0] - 1)

    return len(signal) * summ


def get_signal_data(signal, zones):
    signal_data = list()
    for borders in zones:
        data_part = list()
        for j in range(borders[0], borders[1]):
            data_part.append(signal[j])
        signal_data.append(data_part)
    return signal_data


def get_K(num):
    i = 4
    while num % i != 0:
        i += 1
    return i


def get_Fisher_score(signal, area_data):
    fishers = []
    for i in range(len(area_data)):
        start = area_data[i][0]
        finish = area_data[i][1]
        k = get_K(finish - start)
        while k == finish - start:
            finish += 1
            k = get_K(finish - start)
        fishers.append(get_F(signal[start:finish], int(k)))
    return fishers


def create_areas(signal):
    bin = int(math.log2(len(signal) + 1))
    hist = plt.hist(signal, bins=bin)
    plt.title("Histogram")

    count = []
    start = []
    finish = []
    types = [0] * bin

    for i in range(bin):
        count.append(hist[0][i])
        start.append(hist[1][i])
        finish.append(hist[1][i + 1])

    sortedHist = sorted(count)
    repeat = 0
    for i in range(bin):
        for j in range(bin):
            if sortedHist[len(sortedHist) - 1 - i] == count[j]:
                if repeat == 0:
                    types[j] = "background"
                elif repeat == 1:
                    types[j] = "signal"
                else:
                    types[j] = "transition"
                repeat += 1

    return start, finish, types


def main():
    signal = read_signal_from_file("wave_ampl.txt")
    draw_signal_from_data(signal, 'Signal')
    draw_hist(signal)
    start, finish, types = create_areas(signal)
    zones, zones_types, signal_converted_data = get_converted_data(signal, start, finish, types)
    print("Fisher ", get_Fisher_score(signal, zones))
    draw_areas(signal_converted_data, zones, types, "Separated areas for a signal with no outliers")
    print("zones ", zones)


main()
