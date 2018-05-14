# _*_ coding: utf-8 _*_
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.interpolate as inter
from argparse import ArgumentParser
import os
import numpy as np
import properties as pr
import district_neighbors as dis


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


def visualize(data, m):
    _, ax = plt.subplots(figsize=(10, 10))
    # mapping coordination to data
    dis = -1
    grid = np.zeros(np.shape(m))
    for i, row in enumerate(m):
        for j, col in enumerate(row):          
            dis = col - 1
            if dis >= 0:
                grid[i][j] = data[dis]

    
    grid_ = interpolate(grid)
    # clear_interpolate_bound(grid_, m)
    ax.imshow(grid_, interpolation="bilinear", cmap="viridis")
    ax.set_title("test interpolation")
    plt.show()


# preload map points to matrix map
def build_map():
    m = np.zeros((grid_size, grid_size), dtype=np.int32)
    for k, value in enumerate(dis.points_draw20):
        for part in value:
            corr = part.split(",")
            m[int(corr[1])][int(corr[0])] = k + 1
    return m


grid_size = pr.grid_size
map_ = build_map()
h1 = [19,25,24,16,19,15,12,35,14,26,12,33,11,17,16,16,16,21,14,25,26,22,15,0,18,17]
h2 = [67,78,74,69,54,63,61,45,73,67,53,57,65,73,89,115,64,66,98,52,63,88,49,71,43,35]
# "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"
# seoulmap = mpimg.imread(pr.seoul_map)
# ax.imshow(seoulmap, cmap=plt.cm.gray)
visualize(h1, map_)