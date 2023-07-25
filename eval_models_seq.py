import argparse
import logging
import os
import logging
import torch
from evaluate import *
from util import try_all_gpus
from dataloader_framedeck import Evimo
from swiftnet import SemsegModel, resnet18
from pyramid import SODModel
from recurrentUnet import recurrentUNet_CA_deck, p2t_tiny
from backbones import swin_rnn

model_dir = '/home/shaobo/project/models'
model_dir_voxel = os.path.join(model_dir, 'unet_cs.pth')
model_sod_dir = os.path.join(model_dir, 'sod.pth')
model_swift_dir = os.path.join(model_dir, 'swift.pth')
model_rnn_dir = os.path.join(model_dir, 'R_Unet_ca_2.pth')
model_swin_rnn_dir = os.path.join(model_dir, 'R_swin.pth')
model_swin_dir = os.path.join(model_dir, 'swin.pth')
model_p2t_dir = os.path.join(model_dir, 'p2t.pth')
save_dir = os.path.join('./inference_result_comp')

net_names = ('msrnn',  
            'sodmodel', 'swiftnet', 
            'swin-rnn', 'swin',
            'p2t')

net_dir_dict = {'msrnn':model_rnn_dir, 
            'sodmodel':model_sod_dir, 'swiftnet': model_swift_dir, 
            'swin-rnn': model_swin_rnn_dir, 'swin': model_swin_dir,
            'p2t':model_p2t_dir}

net_dict = {}

def eval_evimo(device, model_rnn, net_1, net_2, model_swin_rnn, model_swin, model_p2t, top_n=200):
    # 加载验证集
    data_module = Evimo(frame_deck=True, n_frame=4)
    data_module.setup()
    # Create data loaders
    val_loader = data_module.val_dataloader()
    # 得到所有类别中最好的结果
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log_dir = os.path.join(save_dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    save_top_n_1f_predict_comp_rnn_labeled(
        val_loader, device, n=top_n, save_dir=save_dir, n_frame=4, model_rnn=model_rnn,
        model_1=net_1, model_2=net_2, model_swin_rnn=model_swin_rnn, model_swin=model_swin, model_p2t=model_p2t)


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--model-size',  type=int,
                        default=1, help='1 for small, 2 for medium, 3 for large')
    parser.add_argument('--classes', '-c', type=int,
                        default=2, help='Number of classes')
    parser.add_argument('--top-n', '-n', type=int,
                        default=20, help='Top n for all classes')
    parser.add_argument('--input-channels', '-i', type=int,
                        default=3, help='Number of input channels')
    parser.add_argument('--gpu', action='store_true',
                        default=False, help='Run with GPU')
    parser.add_argument('--gpu-id', '-g', type=int,
                        default=0, help='Index of gpu you choose')
    parser.add_argument('--n-frame', '-k', type=int,
                        default=4, help='number of frames')
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

    top_n = args.top_n
    net_dict['msrnn'] = recurrentUNet_CA_deck(
        n_channels=args.input_channels, n_classes=args.classes, num_frame=args.n_frame, base_num_channels=32*model_factor,with_ca=args.with_ca)
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

    eval_evimo(model_rnn=net_dict['msrnn'],
               net_1=net_dict['sodmodel'], net_2=net_dict['swiftnet'], 
               model_swin_rnn=net_dict['swin-rnn'], model_swin=net_dict['swin'],
               model_p2t=net_dict['p2t'], device=device,
               top_n=top_n)
