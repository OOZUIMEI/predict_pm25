import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# x = np.linspace(-1,1,100)
# y =  np.linspace(-1,1,100)
# X, Y = np.meshgrid(x,y)

# def f(x, y):
#     s = np.hypot(x, y)
#     phi = np.arctan2(y, x)
#     tau = s + s*(1-s)/5 * np.sin(6*phi) 
#     return 5*(1-tau) + tau

# T = f(X, Y)
# # Choose npts random point from the discrete domain of our model function
# npts = 400
# px, py = np.random.choice(x, npts), np.random.choice(y, npts)
# fig, ax = plt.subplots(nrows=2, ncols=2)
# # Plot the model function and the randomly selected sample points
# ax[0,0].contourf(X, Y, T)
# ax[0,0].scatter(px, py, c='k', alpha=0.2, marker='.')
# ax[0,0].set_title('Sample points on f(X,Y)')

# # Interpolate using three different methods and plot
# for i, method in enumerate(('nearest', 'linear', 'cubic')):
#     a = f(px,py)
#     print(a.shape)
#     Ti = griddata((px, py), a, (X, Y), method=method)
#     r, c = (i+1) // 2, (i+1) % 2
#     ax[r,c].contourf(X, Y, Ti)
#     ax[r,c].set_title('method = {}'.format(method))

# plt.show()
mapx = np.array([0] * 1024)
# idx = [60,184,250,523,525,558,582,583,586,589,770,800,801,818,864,960]
# data = np.array([9.3,17.900000000000002,14.9,13.0,59.0,68.0,16.0,38.0,9.0,1.0,7.0,26.0,8.0,4.0,13.0,66.0])
idx = [344,404,414,439,440,497,500,502,505,523,558,582,583,601,612,613,632,640,641,690,762,770,800,801,818,864,908,960,992]
data = np.array([66.0,111.0,19.0,227.0,120.0,203.0,30.0,67.0,151.0,28.0,75.0,14.0,47.0,39.0,26.0,12.0,93.0,24.0,30.0,49.0,44.0,17.0,35.0,24.5,30.0,23.0,186.0,83.0,70.0])

x = np.linspace(-1,1,32)
y =  np.linspace(-1,1,32)
X, Y = np.meshgrid(x,y)
# py = np.array([y[31 - i / 32] for i in idx])
# px = np.array([x[i % 32]  for i in idx])
py = np.array([y[i / 32] for i in idx])
px = np.array([x[i % 32]  for i in idx])

mapx = np.reshape(mapx, (32,32))
for i, e in enumerate(idx):
    ey = e / 32
    ex = e % 32
    mapx[ey][ex] = data[i]

_, ax = plt.subplots(nrows=1, ncols=2)
ax[0].set_title("original")
ax[0].imshow(mapx, cmap="Oranges")

t = griddata((px, py), data, (X, Y), method="nearest")
ax[1].imshow(t, cmap="Oranges")

plt.show()