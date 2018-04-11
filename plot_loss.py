import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from argparse import ArgumentParser
import os
import math
import utils


def draw(data_plot, colors, lines, title, x_title, y_title, x_custom="", max_val=20, width=640, height=480):
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.set_title(title)
    data = data_plot.split("|")
    data_points = []
    max_val = 0
    min_val = 99999
    for dd in data:
        tmp = [float(dy) for dy in dd.split(",")]
        m = max(tmp)
        min_v = min(tmp)
        if max_val < m:
            max_val = m
        if min_v < min_val:
            min_val = min_v
        data_points.append(tmp)
    max_val = int(math.floor(max_val))
    min_val = int(math.floor(min_val))
    x_sticks = range(len(data_points[0]))
    y_sticks = range(min_val, max_val + 2, 1)
    color = colors.split("|")
    lines = lines.split("|")
    patches = []
    for d, c, l in zip(data_points, color, lines):
        ax.plot(x_sticks, d, color=c)
        d1_patch = mpatches.Patch(color=c, label=l)
        patches.append(d1_patch)
    plt.legend(handles=patches)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_xticks(x_sticks)
    ax.set_yticks(y_sticks)
    ax.set_xticklabels([str((x + 2) * 4) + x_custom for x in x_sticks])
    utils.validate_path("figures/loss")
    plt.savefig("figures/%s.png" % (title.lower().replace(" ", "_")), format="png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", help="file url to plot", default="", type=str)
    parser.add_argument("-t", "--title", help="graph title", default="", type=str)
    parser.add_argument("-w", "--width", help="width", default=640, type=int)
    parser.add_argument("-hi", "--height", help="height", default=480, type=int)
    parser.add_argument("-c", "--colors", default="blue|green")
    parser.add_argument("-l", "--lines", default="blue|green")
    parser.add_argument("-x", "--x", default="Time lags")
    parser.add_argument("-xc", "--xc", default="h")
    parser.add_argument("-y", "--y", default="RMSE")
    parser.add_argument("-m", "--max_val", default=20, type=int)
    
    # title = "RMSE value of Transfer Method Using MSE and MAE Loss"
    # data1 = [12.41,13.48,14.48,14.42,17.30]
    # data2 = [13.73,13.24,13.10,13.30,14.47]
    args = parser.parse_args()
    draw(args.data, args.colors, args.lines, args.title, args.x, args.y, args.xc, args.max_val, args.width, args.height)
    




    