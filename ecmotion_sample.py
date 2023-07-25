import argparse
import logging
import os
import glob
import cv2
import torch
import numpy as np
from util import try_all_gpus
from Voxel import VoxelGrid

real_data_dir = '/mnt2/shaobo/evimo/sunny_record/'
labeled_image_dir = '/mnt2/shaobo/evimo/sunny_record/label_images'
sy_sunny = 262
sx_sunny = 320


def accumulate_image(data, imageH=262, imageW=320, noise_show=False):
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


def get_sunny_aps_all(root_data_dir, video_name):
    """
    根据事件的时间戳窗口范围，返回该窗口中间处时间戳的APS帧
    """
    aps_dir = os.path.join(root_data_dir, 'frames', video_name)
    aps_all = sorted(glob.glob(os.path.join(aps_dir, "*.png")))
    # aps_length = len(aps_all)
    #aps_timestamp = [int(os.path.basename(aps).split('.')[0]) for aps in aps_all]

    return aps_all

# return tensor


def events_to_voxel_accumulate_time_slice(event, channel, np_ts, t0, t1, sensor_size=(262, 320)):
    assert t1 > t0
    if np_ts is None:
        np_ts = event[:, 2]
    start_idx = np.searchsorted(np_ts, t0)
    end_idx = np.searchsorted(np_ts, t1)
    assert start_idx < end_idx
    event = torch.from_numpy(event)  # numpy to torch
    voxel_convertor = VoxelGrid(
        (channel, sensor_size[0], sensor_size[1]), normalize=True)
    event_slice = event[start_idx:end_idx]
    voxel = voxel_convertor.convert(event_slice)
    return voxel


def save_sample(data_name, channel, frame_deck, n_frame, time_slice, device):
    # 得到event data的目录
    event_data_file = os.path.join(real_data_dir, 'events', data_name + '.txt')
    label_data_file = os.path.join(labeled_image_dir, data_name)
    # 将data以aps时间戳的位置取0.03秒的time slice
    event = np.loadtxt(event_data_file)  # (x, y, t, p)
    #event[:, 1] = sy_sunny - event[:, 1] - 1
    ts_all = event[:, 2]
    first_timestamp = event[:, 2][0]
    last_timestamp = event[:, 2][-1]
    # 提取aps图片
    aps_all = get_sunny_aps_all(real_data_dir, data_name)
    # 提取aps图片
    aps_timestamp = []
    # 将time slice处理成voxel grid
    voxel_list = []
    aps_list = []
    label_dict = {}  #创建时间戳与label的映射
    for aps in aps_all:
        timestamp = int(os.path.basename(aps).split('.')[0])
        if timestamp - first_timestamp < time_slice / 2 or last_timestamp - timestamp < time_slice / 2:
            continue
        label_file = os.path.join(label_data_file, str(timestamp) + '.png')
        if os.path.exists(label_file):
            label_image = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            label_image = cv2.threshold(label_image, 30, 255, cv2.THRESH_BINARY)[1]
            label_image = cv2.normalize(
                label_image, label_image, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            label_dict[timestamp] = label_image
        voxel, accu_img = events_to_voxel_accumulate_time_slice(
            event.copy(), channel, ts_all, timestamp - time_slice / 2, timestamp + time_slice / 2)
        voxel_list.append(voxel[:, 3:259, :])
        image = cv2.imread(aps, cv2.IMREAD_GRAYSCALE)  # 读成单通道灰度图

        aps_list.append(image[3:259])
        aps_timestamp.append(timestamp)
       
    if frame_deck is True:
        for i, data in enumerate(voxel_list):
            if i + n_frame < len(voxel_list):
                obj_dict = dict()
                '''
                training = ['box', 'floor', 'table', 'tabletop', 'tabletop-egomotion', 'wall']
                validation = ['box', 'fast', 'floor', 'table', 'tabletop', 'wall']
                '''
                pf = event_data_file.replace('events', 'processed')
                image_deck = [voxel_list[i+j].unsqueeze(0) for j in range(n_frame)]
                mask_deck = [torch.from_numpy(label_dict[aps_timestamp[t]]).unsqueeze(0) for t in range(i, i+ n_frame)]
                gray_deck = [torch.from_numpy(aps_list[i+j]).unsqueeze(0) for j in range(n_frame)]
                images = torch.cat(image_deck, 0)
                masks = torch.cat(mask_deck, 0)
                grays = torch.cat(gray_deck, 0)
                obj_dict['voxel_image'] = images
                obj_dict['gray'] = grays
                obj_dict['mask'] = masks
                os.makedirs(os.path.dirname(pf), exist_ok=True)
                torch.save(obj_dict, pf.replace(".txt",'')+str(i).rjust(5, "0"))


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the model in real-world data!')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--sample', '-s', type=str,
                        default=False, help='eval sample on model')
    parser.add_argument('--model', '-m', type=str,
                        default='unet', help='select model from unet, sod, mdn, ...')
    parser.add_argument('--time-slice',  type=int,
                        default=30000, help='slice of events')
    parser.add_argument('--model-size',  type=int,
                        default=2, help='1 for small, 2 for medium, 3 for large')
    parser.add_argument('--classes', '-c', type=int,
                        default=2, help='Number of classes')
    parser.add_argument('--input-channels', '-i', type=int,
                        default=3, help='Number of input channels')
    parser.add_argument('--gpu', action='store_true',
                        default=False, help='Run with GPU')
    parser.add_argument('--gpu-id', '-g', type=int,
                        default=0, help='Index of gpu you choose')
    parser.add_argument('--n-frame', '-n', type=int,
                        default=4, help='Number of frames for each sample')
    parser.add_argument('--with-ca', action='store_true',
                        default=False, help='with channel wise attention module')
    parser.add_argument('--with-sa', action='store_true',
                        default=False, help='with spatial attention module')

    parser.add_argument('--time-module', action='store_true',
                        default=False, help='unet time with time module')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    if args.gpu is True:
        devices = try_all_gpus()
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = devices[args.gpu_id]
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')
    sample = ('book_1', 'book_2', 'box', 'hdr_walking',
              'w_c_2', 'walk_and_catch')
    #data_name = args.sample
    for data_name in sample:
        save_sample(data_name, channel=3, frame_deck=True, n_frame=4, time_slice=args.time_slice, device=device)