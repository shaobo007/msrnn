import cv2
import argparse
import logging
import os
import logging
import torch
from evaluate_runtime import *
from recurrentUnet import recurrentUNet_CA_deck, p2t_tiny
from backbones import swin_rnn
from util import try_all_gpus
from dataloader_framedeck import Evimo
from pyramid import SODModel
from swiftnet import SemsegModel, resnet18

model_rnn_dir = '/home/shaobo/project/models/R_Unet_0.pth'
model_swin_dir = '/home/shaobo/project/models/swin.pth'
model_swin_rnn_dir = '/home/shaobo/project/models/R_swin.pth'
model_sod_dir = '/home/shaobo/project/models/sod.pth'
model_swift_dir = '/home/shaobo/project/models/swift.pth'
model_p2t_dir = '/home/shaobo/project/models/p2t.pth'
eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')


def eval_runtime(net, device, n_frame, frame_deck=False):
    # 加载验证集
    #data_module = Evimo(frame_deck=True, n_frame=4)
    data_module = Evimo(frame_deck=frame_deck, n_frame=n_frame)
    data_module.setup()
    # Create data loaders
    val_loader = data_module.val_dataloader()
    # 得到所有类别中最好的结果
    #print(evaluate_recurrentUnet_runtime(net, val_loader, device, frame_deck=frame_deck, n_frame=n_frame))
    #print(eval_swift_runtime(net, val_loader, device))
    print(evaluate_sod_runtime(net, val_loader, device))


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--model', '-m', type=str,
                        default='unet', help='select model from unet, sod, mdn, ...')
    parser.add_argument('--model-size',  type=int,
                        default=1, help='1 for small, 2 for medium, 3 for large')
    parser.add_argument('--classes', '-c', type=int,
                        default=2, help='Number of classes')
    parser.add_argument('--top-n', '-n', type=int,
                        default=20, help='Top n for all classes')
    parser.add_argument('--top-N', '-N', type=int,
                        default=10, help='Top n for each classes')
    parser.add_argument('--input-channels', '-i', type=int,
                        default=3, help='Number of input channels')
    parser.add_argument('--gpu', action='store_true',
                        default=False, help='Run with GPU')
    parser.add_argument('--gpu-id', '-g', type=int,
                        default=0, help='Index of gpu you choose')
    parser.add_argument('--n-frame', type=int,
                        default=4, help='Number of frames for each deck')
    parser.add_argument('--with-ca', action='store_true',
                        default=False, help='with channel wise attention module')
    parser.add_argument('--frame-deck', action='store_true',
                        default=False, help='dataset with frame deck')
    parser.add_argument('--with-sa', action='store_true',
                        default=False, help='with spatial attention module')
    parser.add_argument('--with-gray', action='store_true',
                        default=False, help='train and val with grayscale image')
    parser.add_argument('--with-cat', action='store_true',
                        default=False, help='train and val with concatenated image')

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
    model_factor = 1
    if args.model_size == 1:
        model_factor = 0.5
    elif args.model_size == 2:
        model_factor = 1
    elif args.model_size == 3:
        model_factor = 2
    else:
        raise RuntimeError("Wrong model size input !!")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    net_rnn = recurrentUNet_CA_deck(
        n_channels=args.input_channels, n_classes=args.classes, num_frame=4, base_num_channels=int(32*model_factor), with_ca=args.with_ca)
    net_rnn.load_state_dict(torch.load(
        model_rnn_dir, map_location=device))
    net_rnn.to(device=device)
    net_rnn.eval()
    '''
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{net.n_frames} number of frames for each sample\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    '''
    net_sod = SODModel()
    net_sod.load_state_dict(torch.load(
        model_sod_dir, map_location=device))
    net_sod.to(device=device)
    net_sod.eval()
    mean = [73.15, 82.90, 72.3]
    std = [47.67, 48.49, 47.73]
    scale = 1
    backbone = resnet18(pretrained=True, efficient=False,
                        mean=mean, std=std, scale=scale)
    net_swift = SemsegModel(backbone=backbone, num_classes=2)
    net_swift.load_state_dict(torch.load(
        model_swift_dir, map_location=device))
    net_swift.to(device=device)
    net_swift.eval()

    if args.frame_deck:
        net_swin = swin_rnn(args.n_frame)
        net_swin.load_state_dict(torch.load(
            model_swin_rnn_dir, map_location=device))
    else:
        net_swin = swin_rnn(1)
        net_swin.load_state_dict(torch.load(
            model_swin_dir, map_location=device))

    net_swin.to(device=device)
    net_swin.eval()

    net_p2t = p2t_tiny()
    net_p2t.to(device=device)
    net_p2t.eval()

    eval_runtime(net=net_p2t, device=device, n_frame=args.n_frame, frame_deck=args.frame_deck)
               
