# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 21:00
# @Author  : Jiaan Chen

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_point_clouds(data, origin=False, H=260, W=346):
    if not origin:
        # 用于栅格化后的事件可视化
        if data.shape[1] == 3:
            x = data[:, 0]
            y = data[:, 1]
            t_avg = data[:, 2]

            fig = plt.figure('xyt-point cloud')
            ax = Axes3D(fig)
            ax.scatter(x, t_avg, y, marker='.')
            ax.set_xlim3d([0, W])
            ax.set_ylim3d([0, 1])
            ax.set_zlim3d([0, H])
            plt.show()

        elif data.shape[1] == 4:
            x = data[:, 0]
            y = data[:, 1]
            t_avg = data[:, 2]
            p_acc = data[:, 3]
            p_acc -= np.min(p_acc)

            fig = plt.figure('xytp-point cloud')
            color = [plt.get_cmap("Reds", abs(np.max(p_acc) - np.min(p_acc)))(int(i)) for i in p_acc]
            # color = [plt.get_cmap("Reds", abs(np.max(t_avg) - np.min(t_avg)))(int(i)) for i in t_avg]
            ax = Axes3D(fig)
            ax.scatter(x, t_avg, y, c=color, marker='.')
            ax.set_xlim3d([0, W])
            ax.set_ylim3d([0, 1])
            ax.set_zlim3d([0, H])
            # plt.colorbar()  # 创建颜色条，为原比例的0.8
            plt.show()
        elif data.shape[1] == 5:
            x = data[:, 0]
            y = data[:, 1]
            t_avg = data[:, 2]
            p_acc = data[:, 3].copy()
            p_acc[np.where(p_acc == np.min(p_acc))] = np.max(p_acc)
            p_acc -= np.min(p_acc)
            e_cnt = data[:, 4].copy()
            e_cnt[np.where(e_cnt == np.max(e_cnt))] = np.min(e_cnt)

            fig = plt.figure('xytp-point cloud')
            color = [plt.get_cmap("cool", abs(np.max(p_acc) - np.min(p_acc)))(int(i)) for i in p_acc]
            # color = [plt.get_cmap("hot", 256)(int(i)) for i in p_acc]
            size = [20 * i for i in e_cnt]
            ax = Axes3D(fig)
            # ax.grid(False)
            # plt.axis('off')
            ax.scatter(x, t_avg, y, c=color, marker='.', s=size)
            ax.set_xlim3d([0, W])
            ax.set_ylim3d([0, 1])
            ax.set_zlim3d([0, H])
            # plt.colorbar()  # 创建颜色条，为原比例的0.8
            plt.show()
    else:
        # 原始的事件可视化
        x = data[:, 0]
        y = data[:, 1]
        t = data[:, 2]
        p = data[:, 3].copy()  # 原来的数据，p的范围是[0,1]
        p[p == 1] = 60
        # p[0] = 60  # 为了celex数据，保证有一个极性为1，其值设为60，便于可视化

        fig = plt.figure('xyt-point cloud')
        color = [plt.get_cmap("coolwarm", abs(np.max(p) - np.min(p)))(int(i)) for i in p]
        # color = [plt.get_cmap("seismic", 2)(int(i)) for i in p]
        ax = Axes3D(fig)
        # ax.grid(False)
        # plt.axis('off')

        ax.set_xlim3d([0, W])
        ax.set_ylim3d(np.min(t), np.max(t))
        ax.set_zlim3d([0, H])
        ax.scatter(x, t, y, c=color, marker='.')
        plt.show()
