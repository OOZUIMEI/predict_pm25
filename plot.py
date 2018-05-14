from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from argparse import ArgumentParser
import os
import math
import numpy as np
import utils
import properties as pr


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
            # d1 = utils.boost_pm25(d1)
            data1.append(d1)
            data2.append(d2)
            if max_val < d1:
                max_val = d1
            if max_val < d2:
                max_val = d2
    colors = color.split("|")
    max_val = int(math.ceil(max_val * 1.0 / 50) * 50)
    y_sticks = range(0, max_val + 10, 10)
    _, ax = plt.subplots(figsize=(width/100, height/100))
    x = range(len(data1))
    ax.set_title(title)
    ax.plot(x, data2, color=colors[1])
    ax.plot(x, data1, color=colors[0])
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

def decimal_tick(x, pos):
    return '%iK' % (x * 1e-3)


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
    # -t "8h Prediction vs Actual AQI Using,12h Prediction vs Actual AQI,16h Prediction vsActual AQI,20h Prediction vs Actual AQI,24h Prediction vs Actual AQI" -a 1 -c "#dd3913|#0f9617"
    # 4h Prediction on Transfer Method vs Actual AQI,8h Prediction on Transfer Method vs Actual AQI,12h Prediction on Transfer Method vs Actual AQI,16h Prediction on Transfer Method vs Actual AQI,20h Prediction on Transfer Method vs Actual AQI,24h Prediction on Transfer Method vs Actual AQI
    #orange: #ff9901
    #red: #dd3913
    #green: #0f9617
    #blue: #3365cc
    if args.all == 1:
        files = os.listdir(args.file)
        files.sort()
        # print(files)
        if args.title and "," in args.title:
            titles = args.title.split(",")
            if len(files) != len(titles):
                raise ValueError("Not enough titles for files")
        else:
            titles = [str(x) for x in range(1, len(files) + 1)]
        for x, t in zip(files, titles):
            print(x, t)
            plot_prediction("%s/%s" % (args.file, x), t, args.width, args.height, 1, args.color)
    elif args.all == 2:
        # seoul = [536029,1325922,330893,129081,814,407],
        seoul = [0.2307340994,0.5707441547,0.1424331488,0.0555630167,0.0003503869322,0.0001751934661]
        cate = ["Good","Moderate","Unhealthy FSG","Unhealthy","Very Unhealthy","Hazardous"]
        x = range(6)
        # y = [y*1.0 / 10 for y in xrange(1, 10, 2)]
        fig, ax = plt.subplots(figsize=(args.width/100, args.height/100))
        # formatter = FuncFormatter(decimal_tick)
        # ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel("AQI Category")
        ax.set_ylabel("Ratio")
        p1 = mpatches.Patch(color="#1d9667", label="< 50 : Good")
        p2 = mpatches.Patch(color="#fedd47", label="51-100 : Moderate")
        p3 = mpatches.Patch(color="#fd9741", label="101-150 : Unhealthy for Sensitive Group")
        p4 = mpatches.Patch(color="#c8163a", label="151-200 : Unhealthy")
        p5 = mpatches.Patch(color="#641a94", label="201-300 : Very Unhealthy")
        p6 = mpatches.Patch(color="#7c0f29", label="> 300 : Hazardous")
        plt.legend(handles=[p1, p2, p3, p4, p5, p6])
        plt.bar(x, seoul)
        plt.xticks(x, cate, rotation=20)
        # plt.bar(month, over150)
        # plt.bar(month, over200)
        # plt.bar(month, over300)
        # plt.yticks(y)
        plt.savefig("figures/seoul_aqi_ratio.png", format="png", bbox_inches='tight')
    elif args.all == 3:
        bar_width = 0.2
        fig, ax = plt.subplots(figsize=(args.width/100, args.height/100))
        over100 = [39066,32383,39712,32288,32352,26918,25525,13916,12292,18487,25200,32754]
        over150 = [21195,19305,18396,12117,9743,6633,5049,3135,1879,6815,9752,15062]
        over200 = [101,72,62,54,32,114,12,13,7,36,257,54]
        over300 = [44,11,19,18,9,28,16,8,34,30,94,96]
        month = np.arange(1, 13, 1)
        ax.set_xlabel("Month")
        ax.set_ylabel("Ratio (0-1)")
        # ax.tick_params(axis='x', which='major', pad=50)
        s100 = sum(over100)
        s150 = sum(over150)
        s200 = sum(over200)
        s300 = sum(over300)
        over100 = [y*1.0 / s100 for y in over100]
        over150 = [y*1.0 / s150 for y in over150]
        over200 = [y*1.0 / s200 for y in over200]
        over300 = [y*1.0 / s300 for y in over300]
        p1 = ax.bar(month - 0.3, over100, width=bar_width, color='#ff9901')
        p2 = ax.bar(month - 0.1, over150, width=bar_width, color="#dd3913")
        p3 = ax.bar(month + 0.1, over200, width=bar_width, color="#0f9617")
        p4 = ax.bar(month + 0.3, over300, width=bar_width, color="#7c0f29")
        plt.xticks(month, pr.months)
        plt.legend([p1[0], p2[0], p3[0], p4[0]], ("Unhealthy for Sensitive Group","Unhealthy","Very Unhealthy","Hazardous"))
        # plt.show()
        plt.savefig("figures/monthly_ratio.png", format="png", bbox_inches='tight')
    elif args.all == 4:
        bar_width = 0.2
        fig, ax = plt.subplots(figsize=(args.width/100, args.height/100))
        over100 = [60406,51771,58189,44477,42136,33693,30602,17072,14212,25368,35303,47966]
        over150 = [21340,19388,18477,12189,9784,6775,5077,3156,1920,6881,10103,15212]
        month = np.arange(1, 13, 1)
        ax.set_xlabel("Month")
        ax.set_ylabel("Ratio (0-1)")
        # ax.tick_params(axis='x', which='major', pad=50)
        s100 = sum(over100)
        s150 = sum(over150)
        over100 = [y*1.0 / s100 for y in over100]
        over150 = [y*1.0 / s150 for y in over150]
        p1 = ax.bar(month - 0.3, over100, width=bar_width, color='#ff9901')
        p2 = ax.bar(month - 0.1, over150, width=bar_width, color="#0f9617")
        plt.xticks(month, pr.months)
        plt.legend([p1[0], p2[0]], ("> 100","> 150"))
        # plt.show()
        plt.savefig("figures/monthly_ratio_dangerous.png", format="png", bbox_inches='tight')
        
        
    else:
        plot_prediction(args.file, args.title, args.width, args.height, args.save, args.color)