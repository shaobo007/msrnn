# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 21:00
# @Author  : Jiaan Chen
# @File Function:
# ***************************
# SEEM1事件相机数据读取处理和可视化，包含两种方式：
# 1.从全部事件流中按固定事件数量切分成帧并读取，同时读取窗口中间时间戳的APS
# 2.从全部APS帧中读取某一帧及其前后的事件
# 文件夹路径：
# |--root_data_dir
# |-   |--events
# |-   |-   |--video1.txt
# |-   |--frames
# |-   |-   |--video1
# |-   |-   |-   |--frame1.png
# ***************************

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import cv2
from load_tools import get_sunny_events_all, get_sunny_events, get_sunny_aps, accumulate_image, get_sunny_aps_all, get_sunny_APS_add_Events
from plot_tools import plot_point_clouds

sys.path.append("..")

# test sunny data
root_data_dir = r'/home/shaobo/evimo/sunny_record'

sy_sunny = 262
sx_sunny = 320

video_name = 'box'

# **************************
# 根据固定事件数量events_cnt作为一帧进行可视化，展示frame_num处的事件结果，并展示该窗口中央时间戳的APS帧
# **************************

# events_cnt = 7500
# frame_num = 50
#
# video_all, video_length = get_sunny_events_all(root_data_dir, video_name, cnt_event=events_cnt)
# print('video length: {}'.format(video_length))
#
# print('frame number: {}'.format(frame_num))
# data_numpy_origin1, timestamp_range = get_sunny_events(video_all, video_length,
#                                                        frame_num=frame_num, cnt_event=events_cnt)
# print('timestamp_range: {} us'.format(timestamp_range[1] - timestamp_range[0]))
# plot_point_clouds(data_numpy_origin1, origin=True)
#
# data_aps = get_sunny_aps(root_data_dir, video_name, timestamp_range)  # 单通道灰度图[262,320] uint8
#
# image1 = accumulate_image(data_numpy_origin1.T, imageH=sy_sunny, imageW=sx_sunny, noise_show=False)
#
# img_aps = np.concatenate([image1, data_aps], axis=1)
# cv2.imshow('img_aps', img_aps)
# cv2.waitKey(0)


# ****************************
# 根据APS帧，展示该帧前后各一定数量的事件
# ****************************

frame_num = 50

events_cnt = 7500 // 2

events_all, events_len = get_sunny_events_all(root_data_dir, video_name)
aps_all, aps_len = get_sunny_aps_all(root_data_dir, video_name)
print('aps length: {}'.format(aps_len))
aps, data_numpy_origin1 = get_sunny_APS_add_Events(aps_all, events_all, frame_num=frame_num, event_num=events_cnt)
plot_point_clouds(data_numpy_origin1, origin=True)

image1 = accumulate_image(data_numpy_origin1.T, imageH=sy_sunny, imageW=sx_sunny, noise_show=False)
img_aps = np.concatenate([image1, aps], axis=1)
cv2.imshow('img_aps', img_aps)
cv2.waitKey(0)
