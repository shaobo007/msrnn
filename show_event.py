# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import os

def save_point_clouds(data, save_dir, H=260, W=346):
    # 原始的事件可视化
    data[:, 2] = H - data[:, 2] - 1
    x = data[:, 1]
    y = data[:, 2]
    t = data[:, 0]
    p = data[:, 3].copy()  # 原来的数据，p的范围是[0,1]
    p[p == 1] = 60
    # p[0] = 60  # 为了celex数据，保证有一个极性为1，其值设为60，便于可视化

    fig = plt.figure('xyt-point cloud')
    color = [plt.get_cmap("coolwarm", abs(np.max(p) - np.min(p)))(int(i)) for i in p]
    # color = [plt.get_cmap("seismic", 2)(int(i)) for i in p]
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, t, y, c=color, s=1, marker='.')
    ax.grid(False)
    plt.axis('off')

    ax.set_xlim(0, W)
    ax.set_ylim(np.min(t), np.max(t))
    ax.set_zlim(0, H)
    #plt.show()
    plt.savefig(os.path.join(save_dir, 'event.png'), pad_inches=0.1, bbox_inches='tight')
