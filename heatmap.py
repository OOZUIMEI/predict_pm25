# _*_ coding: utf-8 _*_
"""
https://gist.github.com/cpelley/6351152
http://scitools.org.uk/iris/docs/latest/examples/Meteorology/wind_speed.html
Colormap: 
https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
"""
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib.colors  import ListedColormap, BoundaryNorm
import matplotlib.image as mpimg
import scipy.interpolate as inter
from argparse import ArgumentParser
import os
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
def fill_map(data, m, is_interpolate=False, is_clear_out=True):
    data_s = np.shape(data)
    m_s = np.shape(m)
    if len(data_s) is 1:
        grid = np.zeros(m_s)
    else:
        grid = np.zeros(tuple(list(m_s) + [data_s[-1]]))
    dis = -1
    for i, row in enumerate(m):
        for j, col in enumerate(row):          
            dis = col - 1
            if dis >= 0:
                grid[i][j] = data[dis]
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
        grid = dis.points_draw20
    elif grid_size is 30:
        grid = dis.points_30
    elif grid_size is 25:
        grid = dis.points_25
    else:
        raise ValueError("Not support grid size: %i" % grid_size)
    m = np.zeros((grid_size, grid_size), dtype=np.int32)
    for k, value in enumerate(grid):
        for part in value:
            corr = part.split(",")
            m[int(corr[1])][int(corr[0])] = k + 1
    return m



if __name__ == "__main__":
    map_ = build_map()
    pm2_5_range = np.asarray([0, 1, 50, 100, 150, 200, 300, 400, 500], dtype=np.float32) / 500
    bounds = pm2_5_range.tolist()
    cmap = ListedColormap(['grey', 'green', 'yellow', 'orange', 'red', 'purple', 'brown', 'brown'])
    norm = BoundaryNorm(bounds, cmap.N)
    # h1 = [19,25,24,16,19,15,12,35,14,26,12,33,11,17,16,16,16,21,14,25,26,22,15,0,18,17]
    # h2 = np.asarray([67,78,74,69,54,63,61,45,73,67,53,57,65,73,89,115,64,66,98,52,63,88,49,71,43,35])

    # "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"
    # seoulmap = mpimg.imread(pr.seoul_map)
    # ax.imshow(seoulmap, cmap=plt.cm.gray)
    # visualize(h2, map_)
    fig = plt.figure(figsize=(100, 100))
    # data = utils.load_file("vectors/test_sp/sp")
    # data = utils.load_file("vectors/test_sp/non_cnn_grid")
    data = utils.load_file("vectors/test_sp/non_cnn_grid")
    labels = np.asarray(utils.load_file("vectors/test_sp_grid"))
    data = np.asarray(data)
    data = data.reshape([-1])
    data = np.reshape(data, (82 * 128, 24, 25, 25))
    y_s = 9212
    x = data[y_s,:,:,:]
    x = np.where(x >= 0, x, np.zeros(x.shape))
    y = labels[y_s+24:(y_s+48),:,:,0]
    # x = np.where(x >= 0, x, np.zeros(x.shape))
    rows = 6
    cols = 4
    # fig, ax = plt.subplots()
    # cb3 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
    #                             norm=norm,
    #                             boundaries=[-10] + bounds + [10],
    #                             extend='both',
    #                             extendfrac='auto',
    #                             ticks=bounds,
    #                             spacing='uniform',
    #                             orientation='horizontal')
    for i in xrange(0, 11, 2):
        for j in xrange(1, cols + 1):
            ax = fig.add_subplot(rows, cols * 2, i * cols + j)
            ax.set_title("%ih" % i)
            plt.imshow(x[i-1,:,:], cmap=cmap, norm=norm)
    
    for i in xrange(1, 12, 2):
        for j in xrange(1, cols + 1):
            ax_y = fig.add_subplot(rows,  cols * 2, i * cols + j)
            ax_y.set_title("real %ih" % i)
            plt.imshow(y[i-1,:,:], cmap=cmap, norm=norm)
    plt.show()
