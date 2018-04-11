import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from argparse import ArgumentParser
import os
import math
import utils


def plot_prediction(file, title="", width=640, height=480, save=0, color="red|green"):
    with open(file) as f:
        data = f.readlines()
        data1 = []
        data2 = []
        max_val = 100
        for x in data[1:]:
            tmp = x.rstrip("\n").split(",")
            d1 = int(tmp[0])
            d2 = int(tmp[1])
            data1.append(d1)
            data2.append(d2)
            if max_val < d1:
                max_val = d1
            if max_val < d2:
                max_val = d2
    colors = color.split("|")
    max_val = int(math.ceil(max_val * 1.0 / 50) * 50)
    y_sticks = range(0, max_val + 10, 10)
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    x = range(len(data1))
    ax.set_title(title)
    ax.plot(x, data1, color=colors[0])
    ax.plot(x, data2, color=colors[1])
    ax.set_xlabel("Hourly Timestep on Test Set")
    ax.set_ylabel("PM2.5 AQI")
    ax.set_yticks(y_sticks)
    d1_patch = mpatches.Patch(color=colors[0], label="predicted")
    d2_patch = mpatches.Patch(color=colors[1], label="actual")
    plt.legend(handles=[d2_patch, d1_patch])
    if save:
        utils.validate_path("figures/")
        plt.savefig("figures/%s.png" % (title.lower().replace(" ", "_")), format="png", bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a", "--all", help="file url to plot", default=0, type=int)
    parser.add_argument("-f", "--file", help="file url to plot", default="", type=str)
    parser.add_argument("-t", "--title", help="graph title", default="", type=str)
    parser.add_argument("-s", "--save", help="save image or not", default=1, type=int)
    parser.add_argument("-w", "--width", help="width", default=640, type=int)
    parser.add_argument("-hi", "--height", help="height", default=480, type=int)
    parser.add_argument("-c", "--color", default='#dd3913|#0f9617')
    args = parser.parse_args()
    # 4h Prediction on Transfer Method vs Actual AQI,8h Prediction on Transfer Method vs Actual AQI,12h Prediction on Transfer Method vs Actual AQI,16h Prediction on Transfer Method vs Actual AQI,20h Prediction on Transfer Method vs Actual AQI,24h Prediction on Transfer Method vs Actual AQI
    #orange: #ff9901
    #red: #dd3913
    #green: #0f9617
    if args.all:
        files = os.listdir(args.file)
        files.sort()
        print(files)
        if args.title and "," in args.title:
            titles = args.title.split(",")
            if len(files) != len(titles):
                raise ValueError("Not enough titles for files")
        else:
            titles = [str(x) for x in range(1, len(files) + 1)]
        for x, t in zip(files, titles):
            print(x, t)
            plot_prediction("%s/%s" % (args.file, x), t, args.width, args.height, 1, args.color)
    else:
        plot_prediction(args.file, args.title, args.width, args.height, args.save, args.color)