import cv2
import argparse
import logging
import os
import torch
from evaluate import *
from util import try_all_gpus
from dataloader_framedeck import Evimo
from recurrentUnet import recurrentUNet_CA_deck, p2t_tiny
from backbones import swin_rnn

model_dir = '/home/shaobo/project/models'
model_rnn_dir = os.path.join(model_dir, 'R_Unet_0.pth')
model_swin_rnn_dir = os.path.join(model_dir, 'R_swin.pth')
model_swin_dir = os.path.join(model_dir, 'swin.pth')
model_p2t_dir = os.path.join(model_dir, 'p2t.pth')
save_dir = os.path.join('./inference_result_comp')


def eval_evimo(device, model_rnn, frame_deck=False, n_frame=4):
    # 加载验证集
    data_module = Evimo(frame_deck=frame_deck, n_frame=n_frame)
    data_module.setup()
    # Create data loaders
    val_loader = data_module.val_dataloader()
    # 得到所有类别中最好的结果
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log_dir = os.path.join(save_dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    '''
    iou_ratio_1 = eval_acc_exceed_iou_rnn(model_rnn, val_loader, device, 0.6, n_frame=n_frame)
    iou_ratio_2 = eval_acc_exceed_iou_rnn(model_rnn, val_loader, device, 0.5, n_frame=n_frame)
    print('iou > 0.6: ', iou_ratio_1)
    print('iou > 0.5: ', iou_ratio_2)
    '''

    val_score, iou, acc, iou_mean, acc_mean = evaluate_recurrentUnet(model_rnn, val_loader, device,
                                                                       frame_deck=True,
                                                                       n_frame=n_frame, key_idx=0)
    logging.info(
        'val Dice score: {}'.format(val_score))
    logging.info('val IoU: {}'.format(iou))
    logging.info('val Acc: {}'.format(acc))
    logging.info('val mIoU: {}'.format(iou_mean))
    logging.info('val mAcc: {}'.format(acc_mean))

    val_score, iou, acc, iou_mean, acc_mean = evaluate_recurrent_class(
        model_rnn, val_loader, device, n_frame=4)
    logging.info('Validation for each scene')
    logging.info(
        'Validation Dice score: {}'.format(val_score))
    logging.info('Validation IoU: {}'.format(iou))
    logging.info('Validation Acc: {}'.format(acc))
    logging.info('Validation mIoU: {}'.format(iou_mean))
    logging.info('Validation mAcc: {}'.format(acc_mean))
    '''
    eval_iou_deck(net, val_loader, device, save_dir=log_dir,
                  model_time=model_time, model_rnn=model_rnn,
                  model_1=net_1, model_2=net_2)

    save_top_n_1f_predict_comp(
        net, val_loader, device, n=top_n, save_dir=save_dir, n_frame=4, model_rnn=model_rnn,
        model_1=net_1, model_2=net_2)
    save_top_n_seq_predict_comp(net, val_loader, device, top_n, save_dir=save_dir, n_frame=4,
                          model_time=model_time, model_rnn=model_rnn,
                          model_1=net_1, model_2=net_2)
    '''



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
    parser.add_argument('--frame-deck', action='store_true',
                        default=False, help='dataset with frame deck')
    parser.add_argument('--n-frame', '-k', type=int,
                        default=4, help='number of frames')
    parser.add_argument('--with-ca', action='store_true',
                        default=False, help='with channel wise attention module')
    parser.add_argument('--with-sa', action='store_true',
                        default=False, help='with spatial attention module')

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
        model_factor = 1
    elif args.model_size == 2:
        model_factor = 2
    elif args.model_size == 3:
        model_factor = 4
    else:
        raise RuntimeError("Wrong model size input !!")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net_rnn = recurrentUNet_CA_deck(
        n_channels=args.input_channels, n_classes=args.classes, num_frame=args.n_frame, 
        base_num_channels=32*model_factor, with_ca=args.with_ca)
    net_rnn.load_state_dict(torch.load(
        model_rnn_dir, map_location=device))
    net_rnn.to(device=device)
    net_rnn.eval()

    net_swin_rnn = swin_rnn(num_frame=args.n_frame)
    net_swin_rnn.load_state_dict(torch.load(
        model_swin_rnn_dir, map_location=device))
    net_swin_rnn.to(device=device)
    net_swin_rnn.eval()

    net_swin = swin_rnn(num_frame=args.n_frame)
    net_swin.load_state_dict(torch.load(
        model_swin_dir, map_location=device))
    net_swin.to(device=device)
    net_swin.eval()

    net_p2t = p2t_tiny()
    net_p2t.load_state_dict(torch.load(
        model_p2t_dir, map_location=device))
    net_p2t.to(device=device)
    net_p2t.eval()
    eval_evimo(model_rnn=net_rnn,
               device=device,
               frame_deck=args.frame_deck,
               n_frame=args.n_frame
               )
