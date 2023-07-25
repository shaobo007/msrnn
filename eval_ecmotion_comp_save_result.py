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
from Voxel import VoxelGrid, show_voxel_grid
from pyramid import SODModel
from swiftnet import SemsegModel, resnet18
from backbones import swin_rnn
from util import iouEval
from ecmotion_sample import *

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

net_dict = {}


def eval_with_label(data_name, channel, model_no_rnn, model_aps, model_rnn, model_sod, model_swift,
                    model_swin_rnn, model_swin, model_p2t, time_slice, device):
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
    Eval = iouEval(nClasses=2)
    result_txt = f'{data_name}预测结果：\n'
    prev_state_aps = None
    prev_state_voxel = None
    prev_state_swin = None
    label_timestamp_list = []
    for i, timestamp in enumerate(aps_timestamp):
        Eval.reset()
        aps = aps_list[i]
        aps = torch.from_numpy(aps).to(
            device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        voxel = voxel_list[i].to(device).unsqueeze(0)
        batch = {}
        batch['image'] = voxel.clone()
        voxel_image = show_voxel_grid(
            voxel.clone(), sy_image=voxel.shape[2], sx_image=voxel.shape[3])
        with torch.no_grad():
            inference_aps, prev_state_aps = model_aps.sub_forward(
                aps, prev_state_aps)
            inference_rnn, prev_state_voxel = model_rnn.sub_forward(
                voxel.clone(), prev_state_voxel)  # 1 1 h w
            inference_swin_rnn, prev_state_swin = model_swin_rnn.sub_forward(
                voxel.clone(), prev_state_swin)  # 1 1 h w

            if timestamp not in label_dict:
                continue

            # 计算有gt的时间戳的预测
            mask_true = torch.from_numpy(label_dict[timestamp]).to(
                device, dtype=torch.long).unsqueeze(0)
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
            inference_no_rnn = model_no_rnn(voxel.clone())  # 1 1 1 h w
            mask_no_rnn = F.one_hot(inference_no_rnn.round()[0][0].long(), 2).permute(
                0, 3, 1, 2).float()  # 1 2 h w
            Eval.addBatch(mask_no_rnn, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            no_rnn_result = f'unet without rnn iou: {iou[1]} mIoU: {iou_mean}\n'

            Eval.reset()
            mask_aps = F.one_hot(inference_aps.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_aps, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            aps_result = f'aps trained model iou: {iou[1]} mIoU: {iou_mean}\n'

            Eval.reset()
            mask_rnn = F.one_hot(inference_rnn.round()[
                0].long(), 2).permute(0, 3, 1, 2).float()  # n h w
            Eval.addBatch(mask_rnn, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            rnn_result = f'recurrent model iou: {iou[1]} mIoU: {iou_mean}\n'

            inference_sod, _ = model_sod(voxel.clone())
            Eval.reset()
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_result = f'sodmodel iou: {iou[1]} mIoU: {iou_mean}\n'

            logits_swift, _ = model_swift.do_forward(batch, aps.shape[2:4])
            Eval.reset()
            inference_swift = torch.argmax(logits_swift.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_result = f'swiftnet iou: {iou[1]} mIoU: {iou_mean}\n'

            Eval.reset()
            mask_swin_rnn = F.one_hot(inference_swin_rnn.round()[
                0].long(), 2).permute(0, 3, 1, 2).float()  # n h w
            Eval.addBatch(mask_swin_rnn, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swin_rnn_result = f'swin recurrent model iou: {iou[1]} mIoU: {iou_mean}\n'

            Eval.reset()
            inference_swin = model_swin(voxel.clone())  # 1 1 1 h w
            mask_swin = F.one_hot(inference_swin.round()[0][0].long(), 2).permute(
                0, 3, 1, 2).float()  # 1 2 h w
            Eval.addBatch(mask_swin, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swin_result = f'swin without rnn iou: {iou[1]} mIoU: {iou_mean}\n'

            Eval.reset()
            inference_p2t = model_p2t(voxel.clone())
            mask_p2t = F.one_hot(inference_p2t.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_p2t, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            p2t_result = f'sodmodel iou: {iou[1]} mIoU: {iou_mean}\n'


            inference_no_rnn = inference_no_rnn.detach().cpu().round().numpy()
            inference_aps = inference_aps.detach().cpu().round().numpy()
            gray_image = cv2.cvtColor(aps_list[i], cv2.COLOR_GRAY2BGR)

        inference_no_rnn = inference_no_rnn[0][0][0]*255  # h w
        inference_no_rnn = cv2.cvtColor(inference_no_rnn, cv2.COLOR_GRAY2BGR)

        inference_aps = inference_aps[0][0]*255
        inference_aps = cv2.cvtColor(inference_aps, cv2.COLOR_GRAY2BGR)

        inference_rnn = inference_rnn.detach().cpu().round().numpy()  # b 1 h w
        inference_rnn = inference_rnn[0][0]*255
        inference_rnn = cv2.cvtColor(
            inference_rnn, cv2.COLOR_GRAY2BGR)

        inference_sod = inference_sod.detach().cpu().round().numpy()
        inference_sod = inference_sod[0][0]*255
        inference_sod = cv2.cvtColor(inference_sod, cv2.COLOR_GRAY2BGR)

        inference_swift = inference_swift.detach().cpu().numpy()
        inference_swift = inference_swift[0]*255
        inference_swift = cv2.cvtColor(inference_swift, cv2.COLOR_GRAY2BGR)


        inference_swin_rnn = inference_swin_rnn.detach().cpu().round().numpy()  # b 1 h w
        inference_swin_rnn = inference_swin_rnn[0][0]*255
        inference_swin_rnn = cv2.cvtColor(
            inference_swin_rnn, cv2.COLOR_GRAY2BGR)


        inference_swin = inference_swin.detach().cpu().round().numpy()
        inference_swin = inference_swin[0][0][0]*255  # h w
        inference_swin = cv2.cvtColor(inference_swin, cv2.COLOR_GRAY2BGR)

        inference_p2t = inference_p2t.detach().cpu().round().numpy()
        inference_p2t = inference_p2t[0][0]*255
        inference_p2t = cv2.cvtColor(inference_p2t, cv2.COLOR_GRAY2BGR)

        interval = np.full((256, 2, 3), 255)
        '''
        结果图：
        灰度图，voxelgrid可视化，ours， gray_image trained, SODModel, swiftnet
        '''
        gt_mask = label_dict[timestamp]*255
        contours, hierarchy = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 标记掩模轮廓
        gray_image = cv2.drawContours(gray_image, contours, -1, (0, 255, 0), 2)
        gt_image = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
        result = np.concatenate((gray_image, interval, voxel_image,
                                interval, gt_image, interval, inference_rnn, interval, inference_aps, interval,
                                inference_no_rnn, interval, inference_sod, interval, inference_swift, interval,
                                inference_swin_rnn, interval, inference_swin, interval, inference_p2t), axis=1)
        result_txt += f'第{i}张图（时间戳为{timestamp}）预测结果：\n' + \
            rnn_result + aps_result + no_rnn_result + sod_result + swift_result + \
                 swin_rnn_result + swin_result + p2t_result + '\n'
        mask_list.append(result)
        label_timestamp_list.append(timestamp)

    # 得到motion mask
    return mask_list, label_timestamp_list, result_txt


def save_result_label(mask_results, aps_timestamp, result_txt, save_dir):
    save_dir = os.path.join(save_dir, 'inference with label')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'result_com.txt'), 'w') as f:
        f.write(result_txt)
    f.close()
    for i, result in enumerate(mask_results):
        cv2.imwrite(os.path.join(save_dir,  str(
            aps_timestamp[i]) + '_inference.png'), result)


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
        result, aps_timestamp, result_txt = eval_with_label(data_name=data_name, channel=3,
                                                         model_aps=net_dict['msrnn-i'], model_no_rnn=net_dict['msrnn-'],
                                                         model_rnn=net_dict['msrnn'],
                                                         model_sod=net_dict['sodmodel'], model_swift=net_dict['swiftnet'],
                                                         model_swin_rnn=net_dict['swin-rnn'], model_swin=net_dict['swin'],
                                                         model_p2t=net_dict['p2t'],
                                                         time_slice=args.time_slice, device=device)
        save_dir = os.path.join(
            real_data_dir, 'inference_result_comp_deck_rnn_label')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        result_dir = os.path.join(save_dir, data_name)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        save_result_label(result, aps_timestamp, result_txt, result_dir)