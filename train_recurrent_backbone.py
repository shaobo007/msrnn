import argparse
from calendar import c
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import einops
import wandb
import numpy as np
from torch import optim
from tqdm import tqdm
from loss import dice_loss, FocalLoss
from evaluate import evaluate, evaluate_unet, evaluate_recurrentUnet, evaluate_recurrent_class
from util import try_all_gpus
from lr_scheduler import CosineScheduler
from dataloader_framedeck import Evimo
from backbones import recurrent_resnet34, recurrent_resnet18, attention_rnn_56
dir_checkpoint = Path('./checkpoints/')
os.environ["WANDB_API_KEY"] = 'ddf8ed5ce1e799b084c6246cd44eba3dafc3d62c'
os.environ["WANDB_MODE"] = "offline"


class FDLoss(nn.Module):
    def __init__(self):
        super(FDLoss, self).__init__()

    def forward(self, inputs, targets):
        criterion = FocalLoss()
        #raise RuntimeError
        loss = criterion(inputs.float(), targets.float()) \
            + dice_loss(inputs.float(),
                        targets.float())
        return loss


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              n_frame: int = 4,
              frame_deck: bool = False,
              start_epoch: int = 0,
              save_checkpoint: bool = True,
              amp: bool = False):
    # 1. Create dataset
    try:
        data_module = Evimo(batch_size=batch_size,
                            shuffle=True, frame_deck=frame_deck)
        data_module.setup()
    except (AssertionError, RuntimeError):
        print("dataset error!!")

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    n_train = len(train_loader) * batch_size
    n_val = len(val_loader)

    # (Initialize logging)
    experiment = wandb.init(project='rnn-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:            {epochs}
        Batch size:        {batch_size}
        n frames:          {n_frame}
        frame deck:        {frame_deck}
        Learning rate:     {learning_rate}
        Training size:     {n_train}
        Validation size:   {n_val}
        Checkpoints:       {save_checkpoint}
        Device:            {device.type}
        Mixed Precision:   {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    scheduler = CosineScheduler(max_update=10, base_lr=5 * learning_rate, final_lr=0.001 *
                                learning_rate, warmup_steps=5, warmup_begin_lr=learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    Loss_function = FDLoss()
    best_iou = -1 
    best_miou = -1
    best_iou_epoch = start_epoch

    # 5. Begin training
    for epoch in range(start_epoch+1, start_epoch+epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for images, mask_true, _, _ in train_loader:

                # image (b n) c h w
                images = images.to(device=device, dtype=torch.float32)

                true_masks = mask_true.to(
                    device=device, dtype=torch.long)  # true_nasks (b n) h w
                true_masks = einops.rearrange(
                    true_masks, '(b n) h w -> b n h w', n=n_frame)  # b n h w
                #print('true_mask.shape: ', true_masks.shape)

                with torch.cuda.amp.autocast(enabled=amp):

                    masks_pred = net(images)  # masks_pred.shape b n 1 h w
                    true_masks = true_masks.unsqueeze(2)  # b n 1 h w
                    #loss = Loss(criterion, masks_pred, true_mask)
                    loss = Loss_function(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(batch_size)
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        '''
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' +
                                       tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' +
                                       tag] = wandb.Histogram(value.grad.data.cpu())
                        '''
                        val_score, iou, acc, iou_mean, acc_mean = evaluate_recurrentUnet(net, val_loader, device,
                                                                                         frame_deck=frame_deck,
                                                                                         n_frame=n_frame, key_idx=0)
                        if iou.item() > best_iou:
                            best_iou = iou
                            best_iou_epoch = epoch
                            best_miou = iou_mean
                        #val_score, iou, acc, iou_mean, acc_mean = evaluate_unet_1(net, val_loader, device)
                        # scheduler.step(val_score)
                        if scheduler:
                            if scheduler.__module__ == torch.optim.lr_scheduler.__name__:
                                # Using PyTorch In-Built scheduler
                                scheduler.step()
                            else:
                                # Using custom defined scheduler
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = scheduler(epoch)

                        logging.info(
                            'Validation Dice score: {}'.format(val_score))
                        logging.info('Validation IoU: {}'.format(iou))
                        logging.info('Validation Acc: {}'.format(acc))
                        logging.info('Validation mIoU: {}'.format(iou_mean))
                        logging.info('Validation mAcc: {}'.format(acc_mean))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'iou': iou,
                            'acc': acc,
                            'mIoU': iou_mean,
                            'mAcc': acc_mean,
                            'step': global_step,
                            'epoch': epoch,
                            # **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint /
                       'R_resnet_checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    logging.info(f'final best iou = {best_iou} in epoch {best_iou_epoch} best miou = {best_miou}')

    net.load_state_dict(torch.load(str(dir_checkpoint / 'R_resnet_checkpoint_epoch{}.pth'.format(best_iou_epoch)), map_location=device))
    logging.info(f'Model loaded from epoch {best_iou_epoch}')
    val_score, iou, acc, iou_mean, acc_mean = evaluate_recurrentUnet(net, val_loader, device,
                                                                       frame_deck=frame_deck,
                                                                       n_frame=n_frame, key_idx=0)
    logging.info(
        'val Dice score: {}'.format(val_score))
    logging.info('val IoU: {}'.format(iou))
    logging.info('val Acc: {}'.format(acc))
    logging.info('val mIoU: {}'.format(iou_mean))
    logging.info('val mAcc: {}'.format(acc_mean))

    val_score, iou, acc, iou_mean, acc_mean = evaluate_recurrent_class(
        net, val_loader, device, n_frame=n_frame)
    logging.info('Validation for each scene')
    logging.info(
        'Validation Dice score: {}'.format(val_score))
    logging.info('Validation IoU: {}'.format(iou))
    logging.info('Validation Acc: {}'.format(acc))
    logging.info('Validation mIoU: {}'.format(iou_mean))
    logging.info('Validation mAcc: {}'.format(acc_mean))

    logging.info('eval train_loader...')

    Train_score, iou, acc, iou_mean, acc_mean = evaluate_recurrentUnet(net, train_loader, device,
                                                                       frame_deck=frame_deck,
                                                                       n_frame=n_frame, key_idx=0)
    logging.info(
        'Training Dice score: {}'.format(Train_score))
    logging.info('Training IoU: {}'.format(iou))
    logging.info('Training Acc: {}'.format(acc))
    logging.info('Training mIoU: {}'.format(iou_mean))
    logging.info('Training mAcc: {}'.format(acc_mean))
    return best_iou_epoch


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--model-size',  type=int,
                        default=1, help='1 for small, 2 for medium, 3 for large')
    parser.add_argument('--optim', '-o', type=str,
                        default='RMS', help='choose a optimizer to train model')
    parser.add_argument('--scale', '-s', type=float,
                        default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true',
                        default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int,
                        default=1, help='Number of classes')
    parser.add_argument('--n-frame', '-n', type=int,
                        default=4, help='Number of frames for each deck')
    parser.add_argument('--start-epoch', '-v', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--input-channels', '-i', type=int,
                        default=3, help='Number of input channels')
    parser.add_argument('--gpu', action='store_true',
                        default=False, help='Run with GPU')
    parser.add_argument('--gpu-id', '-g', type=int,
                        default=0, help='Index of gpu you choose')
    parser.add_argument('--frame-deck', action='store_true',
                        default=False, help='dataset with frame deck')
    parser.add_argument('--with-ca', action='store_true',
                        default=False, help='with channel wise attention module')

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

    net = recurrent_resnet34(num_frame=args.n_frame)
    #net = attention_rnn_56(num_frame=args.n_frame)
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  n_frame=args.n_frame,
                  frame_deck=args.frame_deck,
                  start_epoch=args.start_epoch,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
