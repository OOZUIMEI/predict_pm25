"""
https://plot.ly/matplotlib/plot/
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from argparse import ArgumentParser
import os
import math
import numpy as np
import utils


def draw(data_points, max_val, min_val, colors, lines, title, x_title, y_title, x_custom="", width=640, height=480, x_interval=20, y_interval=20):
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.set_title(title)
    data_length = len(data_points[0])
    x_points = range(data_length)
    if not x_interval:
        x_interval = 1
    if not y_interval:
        y_interval = 1
    x_sticks = range(0, data_length, x_interval)
    y_sticks = range(min_val, max_val + y_interval, y_interval)
    color = colors.split("|")
    lines = lines.split("|")
    patches = []
    for d, c, l in zip(data_points, color, lines):
        ax.plot(x_points, d, color=c)
        d1_patch = mpatches.Patch(color=c, label=l)
        patches.append(d1_patch)
    plt.legend(handles=patches)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_xticks(x_sticks)
    ax.set_yticks(y_sticks)
    if x_custom:
        ax.set_xticklabels([str((x + 2) * 4) + x_custom for x in x_sticks])
    utils.validate_path("figures")
    if title:
        filename = (title.lower().replace(" ", "_"))
    else:
        filename = "plot_figure"
    plt.savefig("figures/%s.png" % filename, format="png", bbox_inches="tight")
    # plt.show()


def get_min_max(tmp, min_val, max_val):
    m = max(tmp)
    min_v = min(tmp)
    if max_val < m:
        max_val = m
    if min_v < min_val:
        min_val = min_v
    return min_val, max_val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", help="file url to plot", default="", type=str)
    parser.add_argument("-t", "--title", help="graph title", default="", type=str)
    parser.add_argument("-w", "--width", help="width", default=1000, type=int)
    parser.add_argument("-hi", "--height", help="height", default=700, type=int)
    parser.add_argument("-c", "--colors", default="blue|green")
    parser.add_argument("-l", "--lines", default="blue|green")
    parser.add_argument("-x", "--x", default="Time lags")
    parser.add_argument("-xc", "--xc", default="h")
    parser.add_argument("-y", "--y", default="RMSE")
    parser.add_argument("-m", "--min_val", type=int)
    parser.add_argument("-xi", "--x_intervals", default=0, type=int)
    parser.add_argument("-yi", "--y_intervals", default=0, type=int)
    
    # title = "RMSE value of Transfer Method Using MSE and MAE Loss"
    # data1 = [12.41,13.48,14.48,14.42,17.30]
    # data2 = [13.73,13.24,13.10,13.30,14.47]
    # 26.27,27.08,28.61,29.97,31.29|27.51,28.49,29.66,30.6,31.8|29.7,29.65,29.84,30.15,30.79
    args = parser.parse_args()
    max_val = 0
    min_val = 99999
    if "|" not in args.data:
        # it should be a file
        with open(args.data) as file:
            data = [d.rstrip("\n") for d in file.readlines()]
            col_length = len(data[0].split(","))
            data = data[1:]
            row_length = len(data)
            data_points = np.zeros([col_length, row_length])
            for i, d in enumerate(data):
                tmp = [float(dy) for dy in d.split(",")]
                min_val, max_val = get_min_max(tmp, min_val, max_val)
                for x_i, _ in enumerate(data_points):
                    data_points[x_i][i] = tmp[x_i]
    else:
        data_points = []
        data = args.data.split("|")
        for dd in data:
            tmp = [float(dy) for dy in dd.split(",")]
            min_val, max_val = get_min_max(tmp, min_val, max_val)
            data_points.append(tmp)
    max_val = int(math.floor(max_val))
    min_val = int(math.floor(min_val))
    if args.min_val != None:
        min_val = args.min_val
    print(max_val, min_val, len(data_points[0]))
    draw(data_points, max_val, min_val, args.colors, args.lines, args.title, args.x, args.y, args.xc, args.width, args.height, args.x_intervals, args.y_intervals)
    




    