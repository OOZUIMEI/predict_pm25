# _*_ coding: utf-8 _*_
"""
https://gist.github.com/cpelley/6351152
http://scitools.org.uk/iris/docs/latest/examples/Meteorology/wind_speed.html
Colormap: 
https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
https://matplotlib.org/2.0.2/examples/api/colorbar_only.html
"""
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib.colors  import ListedColormap, BoundaryNorm
import matplotlib.image as mpimg
import scipy.interpolate as inter
from colour import Color
from argparse import ArgumentParser
import os
import math
import numpy as np
import properties as pr
import district_neighbors as dis
import utils


def print_map_for_test(grid):
    for x in grid:
        tmp = ""
        for y in x:
            y_s = str(y)
            y_l = len(y_s)
            if y_l == 1:
                y_s += "  "
            elif y_l == 2:
                y_s += " "
            tmp += " %s " % y_s
        print("%s\n" % tmp)


def get_range(st):
    rg = st.split("-")
    fr = int(rg[0])
    to = int(rg[1])
    return fr, to


# interpolate missing point for map
def interpolate(grid):
    shape = np.shape(grid)
    grid_x, grid_y = np.mgrid[0:1:complex(shape[0]), 0:1:complex(shape[1])]
    pts = []
    values = []
    for i, x in enumerate(grid): 
        for j, y in enumerate(x):
            if y:
                values.append(y)
                pts.append([grid_x[i][j], grid_y[i][j]])
    values = np.array(values)
    pts = np.array(pts)
    grid_ = inter.griddata(pts, values, (grid_x, grid_y), method="nearest")
    return grid_


# clear bound of seoul to zeros
def clear_interpolate_bound(grid, m):
    for i, y in enumerate(m):
        j = 0
        for j, c2 in enumerate(y):
            if c2 == 0:
                grid[i][j] = 0


# fill map with normalized pm2.5 value of 25 district
def fill_map(data, districts, is_interpolate=False, is_clear_out=True):
    data_s = np.shape(data)
    if len(data_s) is 1:
        grid = np.zeros((pr.map_size, pr.map_size))
    else:
        # grid_size x grid_size x features_dim
        grid = np.zeros((pr.map_size, pr.map_size, data_s[-1]))
    dis = -1
    for d_i, d in enumerate(districts):
        for p_x, p_y in d:
            grid[p_y][p_x] = data[d_i]
    if is_interpolate:
        grid = interpolate(grid)
        if is_clear_out:
            clear_interpolate_bound(grid, m)
    return grid


# visualize grid_map as heat map
def visualize(data, m):
    _, ax = plt.subplots(figsize=(10, 10))
    # mapping coordination to data
    grid_ = fill_map(data, m)
    ax.imshow(grid_, interpolation="bilinear", cmap="Oranges")
    ax.set_title("Seoul Air Pollution")
    plt.show()


# preload map points to matrix map
def build_map(grid_size=25):
    if grid_size is 60:
        grid = dis.points_draw60
    elif grid_size is 30:
        grid = dis.points_30
    elif grid_size is 25:
        grid = dis.points_25
    else:
        raise ValueError("Not support grid size: %i" % grid_size)
    # m = np.zeros((grid_size, grid_size), dtype=np.int32)
    # for k, value in enumerate(grid):
    #     for part in value:
    #         x, y = part
    #         m[y][x] = k + 1
    return grid


def get_color_map(cr=10):
    pm2_5_aqi = [0, 50, 100, 150, 200, 300, 500]
    b1 = ['#11f52c', '#ffff73', '#ff983d', '#ff312e', '#9000f0', '#851e35', '#690018']
    b2 = ['#009917', '#cfc31d', '#f0690a', '#e00000', '#651a95', '#690018']
    colors = ['gray', '#11f52c']
    pm2_5_range = [0, 0.5]
    for i in xrange(6):
        c = Color(b1[i])
        c_ = Color(b2[i])
        r = int((pm2_5_aqi[i + 1] - pm2_5_aqi[i]) / cr) + 1
        cl = [cl.hex_l for cl in list(c.range_to(c_, r))]
        colors += cl[1:-1] + [b1[i+1]]
        pm = range(pm2_5_aqi[i] + cr, pm2_5_aqi[i + 1] + 1, cr)
        pm2_5_range += pm
    return colors, pm2_5_range


def draw_color_map(cr=10):
    pm2_5_aqi = [0, 50, 100, 150, 200, 300, 500]
    colors, bounds = get_color_map(cr)
    fig = plt.figure(figsize=(6, 20))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    ax2 = fig.add_axes([0.1, 0, 0.1, 0.95])
    cb3 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                boundaries=bounds + [500],
                                extend='both',
                                ticks=bounds,  # optional
                                spacing='proportional',
                                orientation='vertical')
    ticks = [str(int(x + 1)) if x in pm2_5_aqi else "" for x in cb3.get_ticks()]
    ticks = ["0"] + ticks[1:-1] + [500]
    cb3.set_ticklabels(ticks)
    # plt.show()
    plt.savefig("figures/color_bar.png", format="png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    # draw_color_map(5)
    map_ = build_map()
    colors, bounds = get_color_map(5)
    # print(colors)
    # print(bounds)
    fig = plt.figure(figsize=(8, 3))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    # ax2 = fig.add_axes([0.05, 0.15, 0.9, 0.15])
    # cb3 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
    #                             # to use 'extend', you must
    #                             # specify two extra boundaries:
    #                             boundaries=bounds + [500],
    #                             extend='both',
    #                             ticks=bounds,  # optional
    #                             spacing='proportional',
    #                             orientation='horizontal')
    # h1 = [19,25,24,16,19,15,12,35,14,26,12,33,11,17,16,16,16,21,14,25,26,22,15,0,18,17]
    # h2 = np.asarray([67,78,74,69,54,63,61,45,73,67,53,57,65,73,89,115,64,66,98,52,63,88,49,71,43,35])
    # seoulmap = mpimg.imread(pr.seoul_map)
    # ax.imshow(seoulmap, cmap=plt.cm.gray)
    # visualize(h2, map_)
    # fig = plt.figure(figsize=(100, 100), tight_layout=True)
    data = utils.load_file("test_sp/gan_cuda")
    labels = utils.load_file("vectors/test_sp_grid")
    labels = np.asarray(labels)
    grid_sq = int(math.sqrt(data.shape[-1]))
    data = np.reshape(data, (data.shape[0], data.shape[1], grid_sq, grid_sq))
    rows = 6
    cols = 4
    
    for d_i, d in enumerate(data):
        idx = 0
        # x = np.where(d >= 0, d, np.zeros(d.shape))    
        for i in xrange(0, 12, 2):
            for j in xrange(1, cols + 1):
                ax = fig.add_subplot(rows, cols * 2, i * cols + j)
                # ax.set_title("%ih" % (idx + 1))
                plt.imshow(d[idx,:,:] * 300, cmap=cmap, norm=norm)
                idx+= 1
    
        idx = 0
        st = d_i * 4 + 24
        y = labels[st:st + 24,:,:,-1] * 500
        for i in xrange(1, 12, 2):
            for j in xrange(1, cols + 1):
                ax_y = fig.add_subplot(rows,  cols * 2, i * cols + j)
                # ax_y.set_title("real %ih" % (idx + 1))
                plt.imshow(y[idx,:,:], cmap=cmap, norm=norm)
                idx+=1
        # fig.subplots_adjust(top=1.3)
        plt.savefig("figures/gan/%i.png" % d_i, format="png", bbox_inches='tight', dpi=300)


