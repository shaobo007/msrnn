# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 21:00
# @Author  : Jiaan Chen

import os
import numpy as np
import cv2
import glob


def accumulate_image(data, imageH=260, imageW=346, noise_show=False):
    """
    Parameters
    ----------
    data : [3, num_sample]: [x, y, t] or [4, num_sample]: [x, y, t, p]
    Returns
    -------
    image : accumulate image

    """
    x = data[0, :]  # x
    y = data[1, :]  # y
    t = data[2, :]  # t
    if data.shape[0] == 4:
        p = data[3, :]  # p

    img_cam = np.zeros([imageH, imageW])
    num_sample = len(x)

    for idx in range(num_sample):
        # coordx = int(x[idx]) - 1
        # coordy = 260 - int(y[idx]) - 1

        coordx = int(x[idx])  # [0, 345]
        coordy = imageH - int(y[idx]) - 1  # 镜像同时仍然变为[0, 259]

        img_cam[coordy, coordx] = img_cam[coordy, coordx] + 1

    if noise_show:
        img_cam *= 255.0
        image_cam_accumulate = img_cam.astype(np.uint8)
    else:
        image_cam_accumulate = normalizeImage3Sigma(img_cam, imageH, imageW)
        image_cam_accumulate = image_cam_accumulate.astype(np.uint8)

    return image_cam_accumulate


def normalizeImage3Sigma(image, imageH=260, imageW=346):
    """followed by matlab dhp19 generate"""
    sum_img = np.sum(image)
    count_image = np.sum(image > 0)
    mean_image = sum_img / count_image
    var_img = np.var(image[image > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    numSDevs = 3.0
    # Rectify polarity=true
    meanGrey = 0
    range_old = numSDevs * sig_img
    half_range = 0
    range_new = 255
    # Rectify polarity=false
    # meanGrey=127 / 255
    # range= 2*numSDevs * sig_img
    # halfrange = numSDevs * sig_img

    normalizedMat = np.zeros([imageH, imageW])
    for i in range(imageH):
        for j in range(imageW):
            l = image[i, j]
            if l == 0:
                normalizedMat[i, j] = meanGrey
            else:
                f = (l + half_range) * range_new / range_old
                if f > range_new:
                    f = range_new

                if f < 0:
                    f = 0
                normalizedMat[i, j] = np.floor(f)

    return normalizedMat


def get_sunny_events_all(root_data_dir, video_name, cnt_event=7500, sy_sunny=262, sx_sunny=320):
    data = np.loadtxt(root_data_dir + '/events/' + video_name + '.txt')  # 原始数据最小值是0,最大值是319和261
    data[:, 1] = sy_sunny - data[:, 1] - 1  # 调整方向后仍然最小值是0,最大值是319和261

    data_length = data.shape[0]
    video_length = data_length // cnt_event

    return data, video_length


def get_sunny_events(video_all, video_length, cnt_event=7500, frame_num=0):
    if frame_num >= video_length:
        print('Frame number out of range')
        return -1
    frame = video_all[frame_num * cnt_event: (frame_num + 1) * cnt_event, :]
    first_timestamp = frame[:, 2][0]
    last_timestamp = frame[:, 2][-1]
    timestamp_range = [first_timestamp, last_timestamp]

    return frame, timestamp_range


def get_sunny_aps(root_data_dir, video_name, timestamp_range):
    """
    根据事件的时间戳窗口范围，返回该窗口中间处时间戳的APS帧
    """
    aps_dir = root_data_dir + '//frames//' + video_name + '//'
    aps_all = sorted(glob.glob(os.path.join(aps_dir, "*.png")))
    # aps_length = len(aps_all)
    aps_first_timestamp = int(os.path.basename(aps_all[0]).split('.')[0])
    aps_second_timestamp = int(os.path.basename(aps_all[1]).split('.')[0])
    rate = aps_second_timestamp - aps_first_timestamp
    print('aps rate: {}'.format(rate))
    start = np.ceil((timestamp_range[0] - aps_first_timestamp) / rate)  # //除法向下取整
    end = np.floor((timestamp_range[1] - aps_first_timestamp) / rate)
    print('interval aps num: {}'.format((end - start)))
    target = np.floor((end + start) / 2) * rate + aps_first_timestamp  # 返回时间戳范围内中间位置的aps图像
    file_name = aps_dir + str(int(target)) + '.png'
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # 读成单通道灰度图
    print('target aps frame: {}'.format(int(target)))

    return image


def get_sunny_aps_all(root_data_dir, video_name):
    aps_dir = root_data_dir + '//frames//' + video_name + '//'
    aps_all = sorted(glob.glob(os.path.join(aps_dir, "*.png")))
    aps_length = len(aps_all)

    return aps_all, aps_length


def get_sunny_APS_add_Events(aps_all, events_all, frame_num=0, event_num=2000):
    """
    根据当前APS灰度帧的时间戳，查找前后各event_num数量的事件，并返回事件
    """
    aps_first_timestamp = int(os.path.basename(aps_all[0]).split('.')[0])
    aps_second_timestamp = int(os.path.basename(aps_all[1]).split('.')[0])
    rate = aps_second_timestamp - aps_first_timestamp
    print('aps rate: {}'.format(rate))

    aps_file = aps_all[frame_num]
    target_timestamp = int(os.path.basename(aps_file).split('.')[0])
    image = cv2.imread(aps_file, cv2.IMREAD_GRAYSCALE)  # 读成单通道灰度图
    print('target aps frame: {}'.format(target_timestamp))

    mid_events_idx = (np.abs(events_all[:, 2] - target_timestamp)).argmin()
    start_events_idx = max(0, mid_events_idx - event_num // 2)
    end_events_idx = min(len(events_all[:, 0]), mid_events_idx + event_num // 2)
    events_interval = events_all[start_events_idx:end_events_idx, :]
    # events_frame = accumulate_image(events_interval.T, imageH=262, imageW=320)

    return image, events_interval
