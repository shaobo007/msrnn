import argparse
import logging
import os
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from recurrentUnet import recurrentUNet_deck, recurrentUNet_CA_deck, p2t_tiny
from util import try_all_gpus
from pyramid import SODModel
from swiftnet import SemsegModel, resnet18
from backbones import swin_rnn
from util import iouEval
from ecmotion_sample import *
from model_dir import *

'''
model_dir = '/home/shaobo/project/models'
model_dir_no_rnn = os.path.join(model_dir, 'R_Unet_no_rnn.pth')
model_dir_aps = os.path.join(model_dir, 'R_Unet_gray.pth')
model_sod_dir = os.path.join(model_dir, 'sod.pth')
model_swift_dir = os.path.join(model_dir, 'swift.pth')
model_rnn_dir = os.path.join(model_dir, 'R_Unet_ca_2.pth')
model_swin_rnn_dir = os.path.join(model_dir, 'R_swin.pth')
model_swin_dir = os.path.join(model_dir, 'swin.pth')
model_p2t_dir = os.path.join(model_dir, 'p2t.pth')
real_data_dir = '/mnt2/shaobo/evimo/sunny_record/'
labeled_image_dir = '/mnt2/shaobo/evimo/sunny_record/label_images'
'''

sy_sunny = 262
sx_sunny = 320

EVAL_DICT = {}

net_names = ('msrnn', 'msrnn-i', 'msrnn-', 
            'sodmodel', 'swiftnet', 
            'swin-rnn', 'swin', 'p2t')

net_dir_dict = {'msrnn':model_rnn_dir, 'msrnn-i':model_dir_aps, 'msrnn-':model_dir_no_rnn, 
            'sodmodel':model_sod_dir, 'swiftnet': model_swift_dir, 
            'swin-rnn': model_swin_rnn_dir, 'swin': model_swin_dir,
            'p2t':model_p2t_dir}

is_rnn_dict = {'msrnn':True, 'msrnn-i':True, 'msrnn-':False, 
            'sodmodel':False, 'swiftnet': False, 
            'swin-rnn': True, 'swin': False,
            'p2t':False}

net_dict = {}

def eval_iou_with_label_all(data_name, channel, model_dict, time_slice, device):
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
    label_dict = {}  # 创建时间戳与label的映射

    for aps in aps_all:
        timestamp = int(os.path.basename(aps).split('.')[0])
        if timestamp - first_timestamp < time_slice / 2 or last_timestamp - timestamp < time_slice / 2:
            continue
        label_file = os.path.join(label_data_file, str(timestamp) + '.png')
        if os.path.exists(label_file):
            label_image = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            label_image = cv2.threshold(
                label_image, 30, 255, cv2.THRESH_BINARY)[1]
            label_image = cv2.normalize(
                label_image, label_image, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            label_dict[timestamp] = label_image
        voxel = events_to_voxel_accumulate_time_slice(
            event.copy(), channel, ts_all, timestamp - time_slice / 2, timestamp + time_slice / 2)
        voxel_list.append(voxel[:, 3:259, :])
        image = cv2.imread(aps, cv2.IMREAD_GRAYSCALE)  # 读成单通道灰度图

        aps_list.append(image[3:259])
        aps_timestamp.append(timestamp)
    # voxel grid输入至network model当中
    mask_list = []
    prev_state_aps = None
    prev_state_voxel = None
    prev_state_swin = None
    pre_state_dict = {}
    for net_name in net_names:
        if is_rnn_dict[net_name] is True:
            pre_state_dict[net_name] = None

    inference_dict = {}

    for i, timestamp in enumerate(aps_timestamp):
        aps = aps_list[i]
        aps = torch.from_numpy(aps).to(
            device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        voxel = voxel_list[i].to(device).unsqueeze(0)
        batch = {}
        batch['image'] = voxel.clone()
        with torch.no_grad():
            for net_name in net_names:
                if is_rnn_dict[net_name] is True:
                    if net_name == 'msrnn-i':
                        inference_dict[net_name], pre_state_dict[net_name] =\
                        model_dict[net_name].sub_forwad(aps, pre_state_dict[net_name])
                    else:
                        inference_dict[net_name], pre_state_dict[net_name] =\
                        model_dict[net_name].sub_forwad(voxel.clone(), pre_state_dict[net_name])

            inference_aps, prev_state_aps = model_dict['msrnn-i'].sub_forward(
                aps, prev_state_aps)
            inference_rnn, prev_state_voxel = model_dict['msrnn'].sub_forward(
                voxel.clone(), prev_state_voxel)  # 1 1 h w
            inference_swin_rnn, prev_state_swin = model_dict['swin-rnn'].sub_forward(
                voxel.clone(), prev_state_swin)  # 1 1 h w

            if timestamp not in label_dict:
                continue

            # 计算有gt的时间戳的预测
            mask_true = torch.from_numpy(label_dict[timestamp]).to(
                device, dtype=torch.long).unsqueeze(0)
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

            mask_rnn = F.one_hot(inference_rnn.round()[
                0].long(), 2).permute(0, 3, 1, 2).float()  # n h w
            EVAL_DICT[net_names[0]].addBatch(mask_rnn, mask_true)

            mask_aps = F.one_hot(inference_aps.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            EVAL_DICT[net_names[1]].addBatch(mask_aps, mask_true)

            inference_no_rnn = model_dict['msrnn-'](voxel.clone())  # 1 1 1 h w
            mask_no_rnn = F.one_hot(inference_no_rnn.round()[0][0].long(), 2).permute(
                0, 3, 1, 2).float()  # 1 2 h w
            EVAL_DICT[net_names[2]].addBatch(mask_no_rnn, mask_true)

            inference_sod, _ = model_dict['sodmodel'](voxel.clone())
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            EVAL_DICT[net_names[3]].addBatch(mask_sod, mask_true)

            logits_swift, _ = model_dict['swiftnet'].do_forward(batch, aps.shape[2:4])
            inference_swift = torch.argmax(logits_swift.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            EVAL_DICT[net_names[4]].addBatch(mask_swift, mask_true)

            mask_swin_rnn = F.one_hot(inference_swin_rnn.round()[
                0].long(), 2).permute(0, 3, 1, 2).float()  # n h w
            EVAL_DICT[net_names[5]].addBatch(mask_swin_rnn, mask_true)

            inference_swin = model_dict['swin'](voxel.clone())  # 1 1 1 h w
            mask_swin = F.one_hot(inference_swin.round()[0][0].long(), 2).permute(
                0, 3, 1, 2).float()  # 1 2 h w
            EVAL_DICT[net_names[6]].addBatch(mask_swin, mask_true)


            inference_p2t = model_dict['p2t'](voxel.clone())
            mask_p2t = F.one_hot(inference_p2t.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            EVAL_DICT[net_names[7]].addBatch(mask_p2t, mask_true)


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the model in real-world data!')
    parser.add_argument('--time-slice',  type=int,
                        default=30000, help='slice of events')
    parser.add_argument('--model-size',  type=int,
                        default=2, help='1 for small, 2 for medium, 3 for large')
    parser.add_argument('--classes', '-c', type=int,
                        default=1, help='Number of classes')
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
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    if args.gpu is True:
        devices = try_all_gpus()
        device = devices[args.gpu_id]
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')
    model_factor = 1
    if args.model_size == 1:
        model_factor = 1
    elif args.model_size == 2:
        model_factor = 2
    elif args.model_size == 3:
        model_factor = 4
    else:
        raise RuntimeError("Wrong model size input !!")

    net_dict['msrnn'] = recurrentUNet_CA_deck(
        n_channels=args.input_channels, n_classes=args.classes, num_frame=args.n_frame, base_num_channels=32*model_factor,with_ca=args.with_ca)
    net_dict['msrnn-i'] = recurrentUNet_deck(n_channels=1, n_classes=args.classes)
    net_dict['msrnn-'] = recurrentUNet_deck(
        n_channels=args.input_channels, n_classes=args.classes, num_frame=1)
    net_dict['sodmodel'] = SODModel()
    mean = [73.15, 82.90, 72.3]
    std = [47.67, 48.49, 47.73]
    scale = 1
    backbone = resnet18(pretrained=True, efficient=False,
                        mean=mean, std=std, scale=scale)
    net_dict['swiftnet'] = SemsegModel(backbone=backbone, num_classes=2)
    net_dict['swin-rnn'] = swin_rnn(num_frame=4)
    net_dict['swin'] = swin_rnn(num_frame=1)
    net_dict['p2t'] = p2t_tiny()


    for net_name in net_names:
        net_dict[net_name].load_state_dict(torch.load(
        net_dir_dict[net_name], map_location=device))
        net_dict[net_name].to(device=device)
        net_dict[net_name].eval()

    sample = (
            'book_1', 'book_2', 'box',  #proper
            'hdr_walking',              #hdr
            'w_c_2','walk_and_catch'    #poor
            )
    for model in net_names:
        EVAL_DICT[model] = iouEval(nClasses=2)
        EVAL_DICT[model].reset()
    for data_name in sample:
        eval_iou_with_label_all(data_name=data_name, channel=3,
                                model_dict=net_dict,
                                time_slice=args.time_slice, device=device)
    for model in net_names:
        print(f'model {model}')
        iou_mean, iou, _, _ = EVAL_DICT[model].getIoU()
        print(f'iou: {iou[1]}, miou: {iou_mean}')

