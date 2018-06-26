"""
https://matplotlib.org/gallery/images_contours_and_fields/quiver_demo.html#sphx-glr-gallery-images-contours-and-fields-quiver-demo-py
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html
https://math.stackexchange.com/questions/44621/calculate-average-wind-direction
install gpmy2: when missing error come up: apt install libgmp-dev libmpfr-dev libmpc-dev
https://stackoverflow.com/questions/39794338/precise-nth-root
"""

import matplotlib.pyplot as plt
import numpy as np
import heatmap
import math


def min_max_scaler(ws):
    mn = np.min(ws)
    mx = np.max(ws)
    gap = mx - mn
    return (np.array(ws) - mn) / gap


def convert_dir_to_rad(w):
    res = []
    cir = 2 * np.pi 
    for x in w:
        rad = x * np.pi / 180
        t = np.arccos(np.sin(rad))
        if x >= 90 and x <= 270:
            t = cir - t
        res.append(t)
    return res


def avg_wind(w):
    w_ = np.asarray(w)
    w_rad = w_ * math.pi / 180
    u_east = np.mean(np.sin(w_rad))
    u_north = np.mean(np.cos(w_rad))
    unit = np.arctan2(u_east, u_north) * 180 / math.pi
    unit = (360 + unit) % 360
    return unit


def avg_wind2(w):
    w_ = np.asarray(w)
    w_rad = w_ * math.pi / 180
    u_east = np.sum(np.sin(w_rad))
    u_north = np.sum(np.cos(w_rad))
    unit = np.arctan2(u_east, u_north) * 180 / math.pi
    unit = (360 + unit) % 360
    return unit


def visualize(data, m):
    wag, ws = data
    size = np.shape(m)
    X = np.arange(0, size[0], 1)
    grid_x = np.zeros(size)
    grid_y = np.zeros(np.shape(grid_x))
    ra = convert_dir_to_rad(wag)
    U = np.cos(ra)
    V = np.sin(ra)

    ws_n = min_max_scaler(ws)
    U = np.multiply(U, ws_n)
    V = np.multiply(V, ws_n)
    
    _, ax = plt.subplots(figsize=(10, 10))
    # mapping coordination to data
    dis = -1
    
    for i, row in enumerate(m):
        for j, col in enumerate(row):          
            dis = col - 1
            if dis >= 0:
                grid_x[i][j] = U[dis]
                grid_y[i][j] = V[dis]

    # grid_x = interpolate(grid_x)
    # grid_y = interpolate(grid_y)
    # clear_interpolate_bound(grid_, m)
    ax.quiver(X, X, grid_x, grid_y, scale=10, pivot="middle", width=0.001, units="width")
    ax.set_title("Seoul Wind Map")
    plt.show()
    

# map_ = heatmap.build_map()
#""" "NW","W","NW","WSW","W","NNE","NW","NE","WSW","WNW","N","NNW","NNW","NNW","WNW","W","NNW","NW","W","WNW","NNW","WNW","WNW","SW","WSW" """
wag = [24,34,12,12,18,30,23,19,23,37,29,24,50,47,46,57,45,44,26,47,1,3,358,354,321,273,285,345,72,24,323,289,261,326,339,6,343,311,329,348,349,329,339,331,340,323,318,315,314,312,289,304,330,326,328,276,287,340,333]
# wag = [311,264,307,241,266,24,314,48,241,286,0,348,327,327,292,263,331,323,268,285,331,298,288,233,258]
ws = [1.2,1.3,3.3,3.0,2.0,2.3,0.4,0.6,1.3,0.7,0.0,2.5,1.8,2.1,4.9,2.2,3.9,2.3,4.0,2.2,1.3,1.6,0.4,0.3,1.7]
# wgt = [3.1,3.5,4.6,3.6,2.9,2.7,0.5,1.4,1.9,0.9,0.3,3.2,2.5,2.9,7.4,4.8,5.6,3.1,4.8,3.2,1.7,2.3,0.7,0.7,2.4]
# visualize((wag, ws), map_)
print avg_wind2(wag)
# print avg_wind2(wag)