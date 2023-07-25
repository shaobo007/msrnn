import queue
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import einops
from loss import multiclass_dice_coeff, dice_coeff
from util import iouEval
import cv2
import numpy as np
from Voxel import show_voxel_grid, show_voxel_grids
from show_event import save_point_clouds


def evaluate(net, dataloader, device, key_frame=None, n_frame=1, key_idx=0):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        if key_frame is not None:
            mask_true = einops.rearrange(
                mask_true, '(b n) h w -> b n h w', n=n_frame)
            mask_true = mask_true[:, key_idx, :]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(
            device=device, dtype=torch.long).unsqueeze(1).float()
        #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            if n_frame == 1:
                mask_pred, _ = net(image)
            mask_pred = mask_pred.round().float()
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


def evaluate_gvnet(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _, _, gray_image, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        gray_image = gray_image.to(
            device=device, dtype=torch.float32).unsqueeze(1)
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            # predict the mask
            mask_pred = net(gray_image, image)
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


def evaluate_unet_ablation(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for mask_true, gray_image, _, image in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            image = image.to(device=device, dtype=torch.float32)
            mask_pred = net(image)
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


def evaluate_unet_1(net, dataloader, device, with_image=False, with_cat=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            image = image.to(device=device, dtype=torch.float32)
            mask_pred = net(image)
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


def evaluate_sod(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            image = image.to(device=device, dtype=torch.float32)
            mask_pred, _ = net(image)
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


def evaluate_swift(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        mask_true = batch['labels'].to(device)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            logits, additional = net.do_forward(
                batch, batch['labels'].shape[1:3])
            mask_pred = torch.argmax(logits.data, dim=1).float()
            mask_pred = F.one_hot(mask_pred.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


def evaluate_swift_class(net, dataloader, device):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for batch in dataloader:
        # move images and labels to correct device and type
        class_id = batch['class'][0]
        id = validation_class[class_id]
        mask_true = batch['labels'].to(device)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1

        with torch.no_grad():
            # predict the mask
            logits, _ = net.do_forward(batch, batch['labels'].shape[1:3])
            mask_pred = torch.argmax(logits.data, dim=1).float()
            mask_pred = F.one_hot(mask_pred.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean

def save_top_n_1f_predict_comp_rnn_labeled(dataloader, device, n, save_dir, n_frame, model_rnn, model_1, model_2, model_swin_rnn, model_swin, model_p2t):
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    result_queue = queue.PriorityQueue()
    cnt = 0
    for images, mask_true, gray, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        cnt += 1
        batch = {}
        batch['labels'] = mask_true
        gray_image = gray[n_frame - 1][0]
        mask_true = mask_true.to(
            device=device, dtype=torch.long)[n_frame - 1].unsqueeze(0)  # (b n) h w -> n h w
        gts = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            image = images[n_frame - 1].unsqueeze(0)  # 预测最后一帧  1 c h w
            image_copy = image.clone()
            batch['image'] = image.clone()

            inference_rnn = model_rnn(images.clone())  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_rnn = F.one_hot(inference_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_rnn[n_frame - 1].unsqueeze(0), mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            rnn_iou = iou[1]
            if rnn_iou == 0:
                continue

            inference_sod, _ = model_1(image.clone())  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
            Eval.reset()
            inference_swift = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_iou = round(iou[1].item(), 3)

            inference_swin_rnn = model_swin_rnn(images.clone())  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_swin_rnn = F.one_hot(inference_swin_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swin_rnn[n_frame - 1].unsqueeze(0), mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swin_rnn_iou = round(iou[1].item(), 3)

            inference_swin = model_swin(image.clone())  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_swin = F.one_hot(inference_swin.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swin, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swin_iou = round(iou[1].item(), 3)

            inference_p2t = model_p2t(image.clone())  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_p2t = F.one_hot(inference_p2t.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_p2t, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            p2t_iou = round(iou[1].item(), 3)

            mask_true = mask_true.cpu()
            # gray to rgb

            inference_rnn = inference_rnn[0,
                                          n_frame-1].detach().cpu().round().numpy()
            inference_sod = inference_sod.detach().cpu().round().numpy()
            inference_swift = inference_swift.detach().cpu().numpy()
            inference_swin_rnn = inference_swin_rnn[0,
                                          n_frame-1].detach().cpu().round().numpy()
            inference_swin = inference_swin[0,
                                          0].detach().cpu().round().numpy()
            inference_p2t = inference_p2t.detach().cpu().round().numpy()

            gray_image = cv2.cvtColor(
                gray_image.numpy(), cv2.COLOR_GRAY2BGR)

            gt = gts[0].numpy().astype(np.uint8)*255

            contours, hierarchy = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # 标记掩模轮廓
            gray_image = cv2.drawContours(gray_image, contours, -1, (0, 255, 0), 2)

            gt = cv2.cvtColor(
                gt, cv2.COLOR_GRAY2BGR)

            voxel_image = show_voxel_grid(
                image_copy, images.shape[2], images.shape[3])

            inference_rnn_i = inference_rnn[0]*255
            inference_rnn_i = cv2.cvtColor(
                inference_rnn_i, cv2.COLOR_GRAY2BGR)

            inference_sod_i = inference_sod[0][0]*255
            inference_sod_i = cv2.cvtColor(
                inference_sod_i, cv2.COLOR_GRAY2BGR)

            inference_swift_i = inference_swift[0]*255
            inference_swift_i = cv2.cvtColor(
                inference_swift_i, cv2.COLOR_GRAY2BGR)

            inference_swin_rnn_i = inference_swin_rnn[0]*255
            inference_swin_rnn_i = cv2.cvtColor(
                inference_swin_rnn_i, cv2.COLOR_GRAY2BGR)

            inference_swin_i = inference_swin[0]*255
            inference_swin_i = cv2.cvtColor(
                inference_swin_i, cv2.COLOR_GRAY2BGR)

            inference_p2t_i = inference_p2t[0][0]*255
            inference_p2t_i = cv2.cvtColor(
                inference_p2t_i, cv2.COLOR_GRAY2BGR)

            interval_column = np.full((gt.shape[0], 1, 3), 255)

            result = np.concatenate((gray_image, interval_column,
                                     voxel_image, interval_column,
                                     gt, interval_column,
                                     inference_rnn_i, interval_column,
                                     inference_sod_i, interval_column,
                                     inference_swift_i, interval_column,
                                     inference_swin_rnn_i, interval_column,
                                     inference_swin_i, interval_column,
                                     inference_p2t_i, interval_column,
                                     ), axis=1)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            result_iou = str(round(rnn_iou.item(), 3)) + '_' + \
                str(sod_iou) + '_' + str(swift_iou) + '_' + str(swin_rnn_iou) + \
                    '_' + str(swin_iou) + '_' + str(p2t_iou)
            comp_score = rnn_iou.item()
            save_all_dir = os.path.join(save_dir, 'all_1f_predict')
            if not os.path.exists(save_all_dir):
                os.mkdir(save_all_dir)
            cv2.imwrite(os.path.join(os.path.join(save_all_dir, str(
                cnt) + '_' + result_iou + '.png')), result)
            #comp_score = rnn_iou
            result_queue.put((result_iou, result))
            if result_queue.qsize() > n:
                result_queue.get()

    save_top_n_dir = os.path.join(save_dir, 'top_n_1f')
    if not os.path.exists(save_top_n_dir):
        os.mkdir(save_top_n_dir)
    while not result_queue.empty():
        result_iou, result = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(save_top_n_dir, str(
            result_queue.qsize()) + '_' + result_iou + '.png')), result)

def evaluate_AFNet(net, dataloader, device, frame_deck=False, n_frame=2, key_idx=0):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for events, mask_true, images, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        #mask_true = mask_true[:, key_idx, :]
        events = events.to(device=device, dtype=torch.float32)
        images = images.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)  # (b n) h w
        mask_true = einops.rearrange(
            mask_true, '(b n) h w -> b n h w', n=n_frame)  # b n h w
        mask_true = mask_true[:, n_frame - 1]  # last frame # b h w
        mask_true = F.one_hot(mask_true, 2).permute(
            0, 3, 1, 2).float()  # b 1 h w

        with torch.no_grad():
            # predict the mask
            mask_pred = net(events, images)  # b n 1 h w
            mask_pred = mask_pred[:, n_frame - 1]  # b 1 h w

            if net.n_classes == 1:
                mask_pred = F.one_hot(mask_pred.round().squeeze(
                    1).long(), 2).permute(0, 3, 1, 2).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean

def evaluate_recurrentUnet(net, dataloader, device, frame_deck=False, n_frame=2, key_idx=0):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        #mask_true = mask_true[:, key_idx, :]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)  # (b n) h w
        mask_true = einops.rearrange(
            mask_true, '(b n) h w -> b n h w', n=n_frame)  # b n h w
        mask_true = mask_true[:, n_frame - 1]  # last frame # b h w
        mask_true = F.one_hot(mask_true, 2).permute(
            0, 3, 1, 2).float()  # b 1 h w

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)  # b n 1 h w
            mask_pred = mask_pred[:, n_frame - 1]  # b 1 h w

            if net.n_classes == 1:
                mask_pred = F.one_hot(mask_pred.round().squeeze(
                    1).long(), 2).permute(0, 3, 1, 2).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean

def evaluate_recurrentUnet_with_gray(net, dataloader, device, frame_deck=False, n_frame=2, key_idx=0, with_grays=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, grays, in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        #mask_true = mask_true[:, key_idx, :]
        if with_grays:
            images = grays.to(device=device, dtype=torch.float32)
        else:
            images = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)  # (b n) h w
        mask_true = einops.rearrange(
            mask_true, '(b n) h w -> b n h w', n=n_frame)  # b n h w
        mask_true = mask_true[:, n_frame - 1]  # last frame # b h w
        mask_true = F.one_hot(mask_true, 2).permute(
            0, 3, 1, 2).float()  # b 1 h w

        with torch.no_grad():
            # predict the mask
            mask_pred = net(images)  # b n 1 h w
            mask_pred = mask_pred[:, n_frame - 1]  # b 1 h w

            if net.n_classes == 1:
                mask_pred = F.one_hot(mask_pred.round().squeeze(
                    1).long(), 2).permute(0, 3, 1, 2).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean

def evaluate_unet_class(net, dataloader, device, with_image=False, with_cat=False):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, class_id in dataloader:
        # move images and labels to correct device and type
        class_id = class_id[0]
        id = validation_class[class_id]
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1

        with torch.no_grad():
            # predict the mask
            if with_image is True:
                gray_image = gray_image.to(
                    device=device, dtype=torch.float32).unsqueeze(1)
                mask_pred = net(gray_image)
            elif with_cat is True:
                gray_image = gray_image.to(
                    device=device, dtype=torch.float32).unsqueeze(1)
                image = image.to(device=device, dtype=torch.float32)
                input_image = torch.cat([gray_image, image], dim=1)
                mask_pred = net(input_image)
            else:
                image = image.to(device=device, dtype=torch.float32)
                mask_pred = net(image)
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean


def evaluate_unet(net, dataloader, device, frame_deck=False, n_frame=2, key_idx=0):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        if frame_deck is True:
            mask_true = einops.rearrange(
                mask_true, '(b n) h w -> b n h w', n=n_frame)
            mask_true = mask_true[:, key_idx, :]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(
            0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = F.one_hot(mask_pred.round().squeeze(
                    1).long(), 2).permute(0, 3, 1, 2).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


validation_class = {'box': 0, 'fast': 1, 'floor': 2,
                    'table': 3, 'tabletop': 4, 'wall': 5}


def evaluate_unet_3d_pack(net, dataloader, device, frame_deck=False, n_frame=2, key_idx=0):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        if frame_deck is True:
            mask_true = einops.rearrange(
                mask_true, '(b n) h w -> b n h w', n=n_frame)
            mask_true = mask_true[:, key_idx, :]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(
            0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            if frame_deck is True:
                mask_pred = net(image)
            else:
                #mask_pred, _ = net(image)
                mask_pred = net(image)
            # convert to one-hot format
            mask_pred = einops.rearrange(
                mask_pred, '(b n) c h w -> b n c h w', n=n_frame)
            mask_pred = mask_pred[:, key_idx, :]
            if net.n_classes == 1:
                mask_pred = F.one_hot(mask_pred.round().squeeze(
                    1).long(), 2).permute(0, 3, 1, 2).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            Eval.addBatch(mask_pred, mask_true)

    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches, iou[1], acc[1], iou_mean, acc_mean


def evaluate_AFNet_certain_class(net, dataloader, device, save_dir, n_frame=4, cl='box'):
    net.eval()
    cnt = 0
    # iterate over the validation set
    for events, mask_true, gray_image, class_id in dataloader:
        # move images and labels to correct device and type
        class_id = class_id[0]
        if class_id == cl:
            cnt += 1
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = einops.rearrange(
                mask_true, '(b n) h w -> b n h w', n=n_frame)  # b n h w
            mask_true = mask_true[:, n_frame - 1]  # last frame # b h w
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                # predict the mask
                events = events.to(device=device, dtype=torch.float32)
                voxel = show_voxel_grid(events[n_frame - 1].unsqueeze(0).clone(), 256, 336)
                gray_image = gray_image.to(device=device, dtype=torch.float32)
                print(gray_image.shape)
                mask_pred, corr_score, corr_image = net(events, gray_image)
                mask_pred = mask_pred[:, n_frame - 1]  # b 1 1 h w

                mask_pred = mask_pred.squeeze(0)  #1 h w
                corr_score = corr_score.squeeze(0).permute(1, 2, 0)  #3 h w
                corr_image = corr_image.squeeze(0).permute(1, 2, 0)
                inference_afnet = mask_pred.detach().cpu().numpy()
                corr_score = corr_score.detach().cpu().numpy()
                corr_image = corr_image.detach().cpu().numpy()
                gray_image = gray_image.detach().cpu().numpy()
                inference_afnet = inference_afnet[0]*255
                inference_afnet = cv2.cvtColor(inference_afnet, cv2.COLOR_GRAY2BGR)
                corr_score = corr_score*255
                corr_image = corr_image*255
                cv2.imwrite(os.path.join(save_dir, str(cnt) + cl + '_voxel_'  + '.png'), voxel)
                cv2.imwrite(os.path.join(save_dir, str(cnt) + cl + '_inf_'  + '.png'), inference_afnet)
                cv2.imwrite(os.path.join(save_dir, str(cnt) + cl + '_corr_' + '.png'), corr_score)
                cv2.imwrite(os.path.join(save_dir, str(cnt) + cl + '_corr_img_' + '.png'), corr_image)
                cv2.imwrite(os.path.join(save_dir, str(cnt) + cl + '_gray_img_' + '.png'), gray_image[n_frame - 1, 0])

    net.train()

def evaluate_AFNet_class(net, dataloader, device, n_frame=4):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for events, mask_true, gray_image, class_id in dataloader:
        # move images and labels to correct device and type
        class_id = class_id[0]
        id = validation_class[class_id]
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = einops.rearrange(
            mask_true, '(b n) h w -> b n h w', n=n_frame)  # b n h w
        mask_true = mask_true[:, n_frame - 1]  # last frame # b h w
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1

        with torch.no_grad():
            # predict the mask
            events = events.to(device=device, dtype=torch.float32)
            gray_image = gray_image.to(device=device, dtype=torch.float32)
            mask_pred = net(events, gray_image)
            mask_pred = mask_pred[:, n_frame - 1]  # b 1 1 h w
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean
def evaluate_recurrent_class(net, dataloader, device, n_frame=4):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _, class_id in dataloader:
        # move images and labels to correct device and type
        class_id = class_id[0]
        id = validation_class[class_id]
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = einops.rearrange(
            mask_true, '(b n) h w -> b n h w', n=n_frame)  # b n h w
        mask_true = mask_true[:, n_frame - 1]  # last frame # b h w
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1

        with torch.no_grad():
            # predict the mask
            image = image.to(device=device, dtype=torch.float32)
            mask_pred = net(image)
            mask_pred = mask_pred[:, n_frame - 1]  # b 1 1 h w
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean


def evaluate_unet_class(net, dataloader, device, frame_deck=False, n_frame=4, key_idx=0, with_image=False, with_cat=False):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, class_id in dataloader:
        # move images and labels to correct device and type
        if frame_deck is True:
            mask_true = einops.rearrange(
                mask_true, '(b n) h w -> b n h w', n=n_frame)
            mask_true = mask_true[:, key_idx, :]
        class_id = class_id[0]
        id = validation_class[class_id]
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1

        with torch.no_grad():
            # predict the mask
            if with_image is True:
                gray_image = gray_image.to(
                    device=device, dtype=torch.float32).unsqueeze(1)
                mask_pred = net(gray_image)
            elif with_cat is True:
                gray_image = gray_image.to(
                    device=device, dtype=torch.float32).unsqueeze(1)
                image = image.to(device=device, dtype=torch.float32)
                input_image = torch.cat([gray_image, image], dim=1)
                mask_pred = net(input_image)
            else:
                image = image.to(device=device, dtype=torch.float32)
                mask_pred = net(image)
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean


def evaluate_unet_3d_pack_class(net, dataloader, device, frame_deck=False, n_frame=4, key_idx=0):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, class_id in dataloader:
        # move images and labels to correct device and type
        if frame_deck is True:
            mask_true = einops.rearrange(
                mask_true, '(b n) h w -> b n h w', n=n_frame)
            mask_true = mask_true[:, key_idx, :]
        class_id = class_id[0]
        id = validation_class[class_id]
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1

        with torch.no_grad():
            # predict the mask
            image = image.to(device=device, dtype=torch.float32)
            mask_pred = net(image)
            # convert to one-hot format
            mask_pred = einops.rearrange(
                mask_pred, '(b n) c h w -> b n c h w', n=n_frame)
            mask_pred = mask_pred[:, key_idx, :]
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean


def evaluate_sod_class(net, dataloader, device):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _, class_id in dataloader:
        # move images and labels to correct device and type
        class_id = class_id[0]
        id = validation_class[class_id]
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1

        with torch.no_grad():
            # predict the mask
            image = image.to(device=device, dtype=torch.float32)
            mask_pred, _ = net(image)
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean


def evaluate_gvnet_class(net, dataloader, device):
    net.eval()
    num_val_batches = [1] * 6
    dice_score = [0] * 6
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _, _, gray_image, class_id in dataloader:
        # move images and labels to correct device and type
        class_id = class_id[0]
        id = validation_class[class_id]
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        num_val_batches[id] += 1
        gray_image = gray_image.to(
            device=device, dtype=torch.float32).unsqueeze(1)
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            # predict the mask
            mask_pred = net(gray_image, image)
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score[id] += multiclass_dice_coeff(
                mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

            eval_dict[class_id].addBatch(mask_pred, mask_true)

    results = [eval_dict[cl].getIoU() for cl in eval_class]
    iou_mean = [result[0] for result in results]
    IoU = [result[1] for result in results]
    IoU = [iou[1] for iou in IoU]
    acc_mean = [result[2] for result in results]
    Acc = [result[3] for result in results]
    Acc = [acc[1] for acc in Acc]

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return [dice_score[i] / num_val_batches[i] for i in range(6)], IoU, Acc, iou_mean, acc_mean

def eval_acc_exceed_iou_rnn(net, dataloader, device, iou_val, n_frame=4):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    cnt_exceed_iou = 0
    # iterate over the validation set
    for image, mask_true, _, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        mask_true = mask_true.to(device=device, dtype=torch.long)[n_frame-1].unsqueeze(0)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            image = image.to(device=device, dtype=torch.float32)
            mask_pred = net(image)

            mask_pred = F.one_hot(mask_pred.round()[0].squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred[n_frame - 1].unsqueeze(0), mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            if iou[1] > iou_val:
                cnt_exceed_iou += 1

    net.train()

    return cnt_exceed_iou / num_val_batches

def eval_acc_exceed_iou(net, dataloader, device, iou_val, with_image=False, with_cat=False):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    cnt_exceed_iou = 0
    # iterate over the validation set
    for image, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            if with_image is True:
                gray_image = gray_image.to(
                    device=device, dtype=torch.float32).unsqueeze(1)
                mask_pred = net(gray_image)
            elif with_cat is True:
                gray_image = gray_image.to(
                    device=device, dtype=torch.float32).unsqueeze(1)
                image = image.to(device=device, dtype=torch.float32)
                input_image = torch.cat([gray_image, image], dim=1)
                mask_pred = net(input_image)
            else:
                image = image.to(device=device, dtype=torch.float32)
                mask_pred = net(image)

            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            if iou[1] > iou_val:
                cnt_exceed_iou += 1

    net.train()

    return cnt_exceed_iou / num_val_batches


def eval_sod_acc_exceed_iou(net, dataloader, device, iou_val):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    cnt_exceed_iou = 0
    # iterate over the validation set
    for image, mask_true, _, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            image = image.to(device=device, dtype=torch.float32)
            mask_pred, _ = net(image)

            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            if iou[1] > iou_val:
                cnt_exceed_iou += 1

    net.train()

    return cnt_exceed_iou / num_val_batches


def eval_swift_acc_exceed_iou(net, dataloader, device, iou_val):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    cnt_exceed_iou = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        mask_true = batch['labels'].to(device)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            logits, additional = net.do_forward(
                batch, batch['labels'].shape[1:3])
            mask_pred = torch.argmax(logits.data, dim=1).float()
            mask_pred = F.one_hot(mask_pred.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            if iou[1] > iou_val:
                cnt_exceed_iou += 1

    net.train()

    return cnt_exceed_iou / num_val_batches


def eval_gvnet_acc_exceed_iou(net, dataloader, device, iou_val):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    cnt_exceed_iou = 0
    # iterate over the validation set
    for image, mask_true, _, _, gray_image, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        gray_image = gray_image.to(
            device=device, dtype=torch.float32).unsqueeze(1)
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            # predict the mask
            mask_pred = net(gray_image, image)
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            if iou[1] > iou_val:
                cnt_exceed_iou += 1

    net.train()

    return cnt_exceed_iou / num_val_batches


def get_top_n_result(net, dataloader, device, n, save_dir):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    result_queue = queue.PriorityQueue()
    # iterate over the validation set
    for image, mask_true, _, acc_image, gray_image, _, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            inference = mask_pred.cpu()
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            if iou[1].item() == 0:
                continue
            score = 1 - iou[1].item()
            mask_true = mask_true.cpu()
            voxel_image = show_voxel_grid(image)
            # gray to rgb
            gray_image = cv2.cvtColor(
                gray_image[0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference = inference[0][0].round()*255
            inference = cv2.cvtColor(
                inference.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            interval = np.full((260, 2, 3), 255)
            result = np.concatenate(
                (gray_image, interval, voxel_image, interval, gt, interval, inference), axis=1)
            result_queue.put((score, result))
            if result_queue.qsize() > n:
                result_queue.get()

    net.train()
    top_n_dir = os.path.join(save_dir, 'top_n')
    if not os.path.exists(top_n_dir):
        os.mkdir(top_n_dir)
    while not result_queue.empty():
        score, result = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(top_n_dir, str(
            result_queue.qsize()) + str(1 - score) + '.png')), result)
        '''
        file_dir = os.path.join(top_n_dir, str(result_queue.qsize()) + str(score))
        if not os.path.exists(file_dir):
            os.mkdir(os.path.join(file_dir))
        cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0].numpy())
        cv2.imwrite(os.path.join(file_dir,  'voxel.png'), result[1])
        cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2].numpy())
        cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3].numpy())
        '''

    return result_queue


def get_top_n_result_comp(net, dataloader, device, n, save_dir, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    result_queue = queue.PriorityQueue()
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        image = batch['image'].to(device=device, dtype=torch.float32)
        mask_true = batch['labels'].to(device)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred_0 = net(image)
            inference_0 = mask_pred_0.cpu()
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred_0, mask_true)

            mask_pred_1, _ = model_1(image)
            inference_1 = mask_pred_1.cpu()

            logits, _ = model_2.do_forward(batch, batch['labels'].shape[1:3])
            inference_2 = torch.argmax(logits.data, dim=1).float().cpu()

            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            if iou[1].item() == 0:
                continue
            score = iou[1]
            mask_true = mask_true.cpu()
            voxel_image = show_voxel_grid(
                image, image.shape[2], image.shape[3])
            # gray to rgb
            gray_image = cv2.cvtColor(
                batch['gray'][0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference_0 = inference_0[0][0].round()*255
            inference_0 = cv2.cvtColor(
                inference_0.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

            inference_1 = inference_1[0][0].round()*255
            inference_1 = cv2.cvtColor(
                inference_1.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

            inference_2 = inference_2[0]*255
            inference_2 = cv2.cvtColor(
                inference_2.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            interval = np.full((256, 2, 3), 255)
            result = np.concatenate((gray_image, interval, voxel_image, interval, gt,
                                    interval, inference_0, interval, inference_1, interval, inference_2), axis=1)
            result_queue.put((score, result))
            if result_queue.qsize() > n:
                result_queue.get()

    net.train()
    top_n_dir = os.path.join(save_dir, 'top_n')
    if not os.path.exists(top_n_dir):
        os.mkdir(top_n_dir)
    while not result_queue.empty():
        score, result = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(top_n_dir, str(
            result_queue.qsize()) + str(score) + '.png')), result)
        '''
        file_dir = os.path.join(top_n_dir, str(result_queue.qsize()) + str(score))
        if not os.path.exists(file_dir):
            os.mkdir(os.path.join(file_dir))
        cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0].numpy())
        cv2.imwrite(os.path.join(file_dir,  'voxel.png'), result[1])
        cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2].numpy())
        cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3].numpy())
        '''

    return result_queue


def get_top_n_result_iou_comp(net, dataloader, device, n, save_dir, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    result_queue = queue.PriorityQueue()
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        image = batch['image'].to(device=device, dtype=torch.float32)
        mask_true = batch['labels'].to(device)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            Eval.reset()
            # predict the mask
            mask_pred_0 = net(image)
            inference_0 = mask_pred_0.cpu()
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred_0, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            our_iou = iou[1]
            if our_iou == 0:
                continue
            score = our_iou
            inference_1, _ = model_1(image)
            Eval.reset()
            mask_sod = F.one_hot(inference_1.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(batch, batch['labels'].shape[1:3])
            Eval.reset()
            inference_2 = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_2.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_iou = round(iou[1].item(), 3)
            mask_true = mask_true.cpu()
            voxel_image = show_voxel_grid(
                image, image.shape[2], image.shape[3])
            # gray to rgb
            gray_image = cv2.cvtColor(
                batch['gray'][0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference_0 = inference_0.detach().cpu().round().numpy()
            inference_0 = inference_0[0][0]*255
            inference_0 = cv2.cvtColor(
                inference_0, cv2.COLOR_GRAY2BGR)

            inference_1 = inference_1.detach().cpu().round().numpy()
            inference_1 = inference_1[0][0]*255
            inference_1 = cv2.cvtColor(
                inference_1, cv2.COLOR_GRAY2BGR)

            inference_2 = inference_2.detach().cpu().numpy()
            inference_2 = inference_2[0]*255
            inference_2 = cv2.cvtColor(
                inference_2, cv2.COLOR_GRAY2BGR)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            interval = np.full((256, 2, 3), 255)
            result = np.concatenate((gray_image, interval, voxel_image, interval, gt,
                                    interval, inference_0, interval, inference_1, interval, inference_2), axis=1)
            result_iou = str(round(our_iou.item(), 3)) + '_' + \
                str(sod_iou) + '_' + str(swift_iou)
            result_queue.put((score, result, result_iou))
            if result_queue.qsize() > n:
                result_queue.get()

    net.train()
    top_n_dir = os.path.join(save_dir, 'top_n')
    if not os.path.exists(top_n_dir):
        os.mkdir(top_n_dir)
    while not result_queue.empty():
        score, result, result_iou = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(top_n_dir, str(
            result_queue.qsize()) + '_' + result_iou + '.png')), result)
        '''
        file_dir = os.path.join(top_n_dir, str(result_queue.qsize()) + str(score))
        if not os.path.exists(file_dir):
            os.mkdir(os.path.join(file_dir))
        cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0].numpy())
        cv2.imwrite(os.path.join(file_dir,  'voxel.png'), result[1])
        cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2].numpy())
        cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3].numpy())
        '''

    return result_queue


def get_top_n_result_for_each_class_comp(net, dataloader, device, n, save_dir, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    queue_dict = {}
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        queue_dict[cl] = queue.PriorityQueue()
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        class_id = batch['class'][0]
        image = batch['image'].to(device)
        mask_true = batch['labels'].to(device)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            eval_dict[class_id].reset()
            mask_pred_0 = net(image)
            inference_0 = mask_pred_0.cpu()
            # convert to one-hot format
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background

            mask_pred_1, _ = model_1(image)
            inference_1 = mask_pred_1.cpu()

            logits, _ = model_2.do_forward(batch, batch['labels'].shape[1:3])
            inference_2 = torch.argmax(logits.data, dim=1).float().cpu()

            eval_dict[class_id].addBatch(mask_pred_0, mask_true)
            iou_mean, iou, acc_mean, acc = eval_dict[class_id].getIoU()
            if iou[1].item() == 0:
                continue
            score = iou[1]
            voxel_image = show_voxel_grid(
                image, image.shape[2], image.shape[3])
            gray_image = cv2.cvtColor(
                batch['gray'][0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference_0 = inference_0[0][0].round()*255
            inference_0 = cv2.cvtColor(
                inference_0.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

            inference_1 = inference_1[0][0].round()*255
            inference_1 = cv2.cvtColor(
                inference_1.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)

            inference_2 = inference_2[0]*255
            inference_2 = cv2.cvtColor(
                inference_2.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            interval = np.full((256, 2, 3), 255)
            result = np.concatenate((gray_image, interval, voxel_image, interval, gt,
                                    interval, inference_0, interval, inference_1, interval, inference_2), axis=1)
            queue_dict[class_id].put((score, result))
            if queue_dict[class_id].qsize() > n:
                queue_dict[class_id].get()

    for cl in eval_class:
        class_dir = os.path.join(save_dir, cl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        while not queue_dict[cl].empty():
            score, result = queue_dict[cl].get()
            cv2.imwrite(os.path.join(os.path.join(class_dir, str(
                queue_dict[cl].qsize()) + str(score) + '.png')), result)
            '''
            file_dir = os.path.join(class_dir, str(queue_dict[cl].qsize()) + str(score))
            if not os.path.exists(os.path.join(file_dir)):
                os.mkdir(os.path.join(file_dir))
            cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0].numpy())
            cv2.imwrite(os.path.join(file_dir,  'voxel.png'), result[1])
            cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2].numpy())
            cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3].numpy())
            '''
    net.train()
    return queue_dict


def get_top_n_result_for_each_class_iou_comp(net, dataloader, device, n, save_dir, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    queue_dict = {}
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        queue_dict[cl] = queue.PriorityQueue()
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        class_id = batch['class'][0]
        image = batch['image'].to(device)
        mask_true = batch['labels'].to(device)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            eval_dict[class_id].reset()
            mask_pred_0 = net(image)
            inference_0 = mask_pred_0.cpu()
            # convert to one-hot format
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            eval_dict[class_id].addBatch(mask_pred_0, mask_true)
            iou_mean, iou, acc_mean, acc = eval_dict[class_id].getIoU()
            our_iou = iou[1]
            if our_iou == 0:
                continue
            score = iou[1]

            inference_1, _ = model_1(image)
            eval_dict[class_id].reset()
            mask_sod = F.one_hot(inference_1.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            eval_dict[class_id].addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = eval_dict[class_id].getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(batch, batch['labels'].shape[1:3])
            eval_dict[class_id].reset()
            inference_2 = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_2.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            eval_dict[class_id].addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = eval_dict[class_id].getIoU()
            swift_iou = round(iou[1].item(), 3)

            voxel_image = show_voxel_grid(
                image, image.shape[2], image.shape[3])
            gray_image = cv2.cvtColor(
                batch['gray'][0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference_0 = inference_0.detach().cpu().round().numpy()
            inference_0 = inference_0[0][0]*255
            inference_0 = cv2.cvtColor(
                inference_0, cv2.COLOR_GRAY2BGR)

            inference_1 = inference_1.detach().cpu().round().numpy()
            inference_1 = inference_1[0][0]*255
            inference_1 = cv2.cvtColor(
                inference_1, cv2.COLOR_GRAY2BGR)

            inference_2 = inference_2.detach().cpu().numpy()
            inference_2 = inference_2[0]*255
            inference_2 = cv2.cvtColor(
                inference_2, cv2.COLOR_GRAY2BGR)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            interval = np.full((256, 2, 3), 255)
            result = np.concatenate((gray_image, interval, voxel_image, interval, gt,
                                    interval, inference_0, interval, inference_1, interval, inference_2), axis=1)
            result_iou = str(round(our_iou.item(), 3)) + '_' + \
                str(sod_iou) + '_' + str(swift_iou)
            queue_dict[class_id].put((score, result, result_iou))
            if queue_dict[class_id].qsize() > n:
                queue_dict[class_id].get()

    for cl in eval_class:
        class_dir = os.path.join(save_dir, cl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        while not queue_dict[cl].empty():
            score, result = queue_dict[cl].get()
            cv2.imwrite(os.path.join(os.path.join(class_dir, str(
                queue_dict[cl].qsize()) + '_' + result_iou + '.png')), result)
            '''
            file_dir = os.path.join(class_dir, str(queue_dict[cl].qsize()) + str(score))
            if not os.path.exists(os.path.join(file_dir)):
                os.mkdir(os.path.join(file_dir))
            cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0].numpy())
            cv2.imwrite(os.path.join(file_dir,  'voxel.png'), result[1])
            cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2].numpy())
            cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3].numpy())
            '''
    net.train()
    return queue_dict


def get_top_n_result_for_each_class(net, dataloader, device, n, save_dir):
    net.eval()
    num_val_batches = len(dataloader)
    queue_dict = {}
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    for cl in eval_class:
        queue_dict[cl] = queue.PriorityQueue()
        eval_dict[cl] = iouEval(nClasses=2)
    # iterate over the validation set
    for image, mask_true, _, acc_image, gray_image, class_id, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        class_id = class_id[0]
        eval_dict[class_id].reset()
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            inference = mask_pred.cpu()
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background

            eval_dict[class_id].addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = eval_dict[class_id].getIoU()
            if iou[1].item() == 0:
                continue
            score = 1 - iou[1].item()
            voxel_image = show_voxel_grid(image)
            gray_image = cv2.cvtColor(
                gray_image[0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference = inference[0][0].round()*255
            inference = cv2.cvtColor(
                inference.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            interval = np.full((260, 2, 3), 255)
            result = np.concatenate(
                (gray_image, interval, voxel_image, interval, gt, interval, inference), axis=1)
            queue_dict[class_id].put((score, result))
            if queue_dict[class_id].qsize() > n:
                queue_dict[class_id].get()

    for cl in eval_class:
        class_dir = os.path.join(save_dir, cl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        while not queue_dict[cl].empty():
            score, result = queue_dict[cl].get()
            cv2.imwrite(os.path.join(os.path.join(class_dir, str(
                queue_dict[cl].qsize()) + str(1 - score) + '.png')), result)
            '''
            file_dir = os.path.join(class_dir, str(queue_dict[cl].qsize()) + str(score))
            if not os.path.exists(os.path.join(file_dir)):
                os.mkdir(os.path.join(file_dir))
            cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0].numpy())
            cv2.imwrite(os.path.join(file_dir,  'voxel.png'), result[1])
            cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2].numpy())
            cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3].numpy())
            '''
    net.train()
    return queue_dict


def get_top_n_result_each_file(net, dataloader, device, n, save_dir):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    result_queue = queue.PriorityQueue()
    # iterate over the validation set
    for image, mask_true, _, acc_image, gray_image, _, event in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        Eval.reset()
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            inference = mask_pred.cpu()
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            score = iou[1]
            mask_true = mask_true.cpu()
            voxel_image = show_voxel_grids(image)
            # gray to rgb
            gray_image = cv2.cvtColor(
                gray_image[0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference = inference[0][0].round()*255
            inference = cv2.cvtColor(
                inference.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            result = [gray_image, voxel_image, gt,
                      inference, event[0]]
            #interval = np.full((260, 2, 3), 255)
            #result = np.concatenate((gray_image, interval, voxel_image, interval, gt, interval, inference), axis=1)
            result_queue.put((score, result))
            if result_queue.qsize() > n:
                result_queue.get()

    net.train()
    top_n_dir = os.path.join(save_dir, 'top_n')
    if not os.path.exists(top_n_dir):
        os.mkdir(top_n_dir)
    while not result_queue.empty():
        score, result = result_queue.get()
        file_dir = os.path.join(top_n_dir, str(
            result_queue.qsize()) + str(score))
        if not os.path.exists(file_dir):
            os.mkdir(os.path.join(file_dir))
        cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0])
        cv2.imwrite(os.path.join(file_dir,  'voxel_0.png'), result[1][0])
        cv2.imwrite(os.path.join(file_dir,  'voxel_1.png'), result[1][1])
        cv2.imwrite(os.path.join(file_dir,  'voxel_2.png'), result[1][2])
        cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2])
        cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3])
        save_point_clouds(result[4].numpy(), file_dir)

    return result_queue


def get_top_n_result_for_each_class_each_file(net, dataloader, device, n, save_dir):
    net.eval()
    num_val_batches = len(dataloader)
    queue_dict = {}
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    eval_dict = {}
    class_cnt = {}
    for cl in eval_class:
        queue_dict[cl] = queue.PriorityQueue()
        eval_dict[cl] = iouEval(nClasses=2)
        class_cnt[cl] = 0
    # iterate over the validation set
    for image, mask_true, _, acc_image, gray_image, class_id, event in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        class_id = class_id[0]
        eval_dict[class_id].reset()
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        gt = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            inference = mask_pred.cpu()
            # convert to one-hot format
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background

            eval_dict[class_id].addBatch(mask_pred, mask_true)
            iou_mean, iou, acc_mean, acc = eval_dict[class_id].getIoU()
            score = iou[1]
            voxel_image = show_voxel_grids(image)
            gray_image = cv2.cvtColor(
                gray_image[0].numpy(), cv2.COLOR_GRAY2BGR)
            gt = gt[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            inference = inference[0][0].round()*255
            inference = cv2.cvtColor(
                inference.numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
            result = [gray_image, voxel_image, gt,
                      inference, event[0]]
            #interval = np.full((260, 2, 3), 255)
            #result = np.concatenate((gray_image, interval, voxel_image, interval, gt, interval, inference), axis=1)
            queue_dict[class_id].put((score, result))
            class_cnt[class_id] += 1
            if queue_dict[class_id].qsize() > n:
                queue_dict[class_id].get()

    for cl in eval_class:
        class_dir = os.path.join(save_dir, cl)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        print(cl, 'has ', class_cnt[cl], 'samples')
        while not queue_dict[cl].empty():
            score, result = queue_dict[cl].get()
            #cv2.imwrite(os.path.join(os.path.join(class_dir, str(queue_dict[cl].qsize()) + str(score) + '.png')), result)
            file_dir = os.path.join(class_dir, str(
                queue_dict[cl].qsize()) + str(score))
            if not os.path.exists(os.path.join(file_dir)):
                os.mkdir(os.path.join(file_dir))
            cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0])
            cv2.imwrite(os.path.join(file_dir,  'voxel_0.png'), result[1][0])
            cv2.imwrite(os.path.join(file_dir,  'voxel_1.png'), result[1][1])
            cv2.imwrite(os.path.join(file_dir,  'voxel_2.png'), result[1][2])
            cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2])
            cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3])
            save_point_clouds(result[4].numpy(), file_dir)
    net.train()
    return queue_dict


def eval_model(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    eval_class = ('box', 'fast', 'floor', 'table', 'tabletop', 'wall')
    class_cnt = {}
    for cl in eval_class:
        class_cnt[cl] = 0
    # iterate over the validation set
    for image, mask_true, _, acc_image, gray_image, class_id, event in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        class_id = class_id[0]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            mask_pred = net(image)
            inference = mask_pred.cpu()
            mask_pred = F.one_hot(mask_pred.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()

            inference = inference[0][0].round()*255
            inference = inference.numpy().astype(np.uint8)
            class_cnt[class_id] += 1
    for cl in eval_class:
        print(cl, 'has ', class_cnt[cl], 'samples')
    net.train()

# 测试相邻帧的预测区别


def get_top_n_result_iou_comp_deck(net, dataloader, device, n, save_dir, n_frame, model_time, model_rnn, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    result_queue = queue.PriorityQueue()
    # iterate over the validation set
    for images, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        batch = {}
        batch['labels'] = mask_true
        mask_true = mask_true.to(
            device=device, dtype=torch.long)  # (b n) h w -> n h w
        gts = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            Eval.reset()
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            batch['image'] = images
            mask_pred_0 = net(images)   # (b n) 1 h w
            inference_0 = mask_pred_0.cpu()
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred_0, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            our_iou = iou[1]
            if our_iou == 0:
                continue

            inference_time = model_time(images)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_time = F.one_hot(inference_time.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_time, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            time_iou = round(iou[1].item(), 3)

            inference_rnn = model_rnn(images)  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_rnn = F.one_hot(inference_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_rnn, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            rnn_iou = round(iou[1].item(), 3)

            inference_sod, _ = model_1(images)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
            Eval.reset()
            inference_swift = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_iou = round(iou[1].item(), 3)
            mask_true = mask_true.cpu()
            # gray to rgb

            voxel_list = []
            gt_list = []
            inferences_0 = []
            inferences_time = []
            inferences_rnn = []
            inferences_sod = []
            inferences_swift = []

            inference_0 = inference_0.detach().cpu().round().numpy()
            inference_sod = inference_sod.detach().cpu().round().numpy()
            inference_time = inference_time.detach().cpu().round().numpy()
            inference_rnn = inference_rnn[0, :].detach().cpu().round().numpy()
            inference_swift = inference_swift.detach().cpu().numpy()

            #interval = np.full((256, 1, 3), 255)
            interval_row = np.full((1, 336, 3), 255)

            for i in range(n_frame):
                gt = gts[i]*255
                gt = cv2.cvtColor(gt.numpy().astype(
                    np.uint8), cv2.COLOR_GRAY2BGR)
                gt_list.append(gt)
                gt_list.append(interval_row)

                voxel_image = show_voxel_grid(
                    images[i].unsqueeze(0), images.shape[2], images.shape[3])
                voxel_list.append(voxel_image)
                voxel_list.append(interval_row)

                inference_0_i = inference_0[i][0]*255
                inference_0_i = cv2.cvtColor(
                    inference_0_i, cv2.COLOR_GRAY2BGR)
                inferences_0.append(inference_0_i)
                inferences_0.append(interval_row)

                inference_sod_i = inference_sod[i][0]*255
                inference_sod_i = cv2.cvtColor(
                    inference_sod_i, cv2.COLOR_GRAY2BGR)
                inferences_sod.append(inference_sod_i)
                inferences_sod.append(interval_row)

                inference_time_i = inference_time[i][0]*255
                inference_time_i = cv2.cvtColor(
                    inference_time_i, cv2.COLOR_GRAY2BGR)
                inferences_time.append(inference_time_i)
                inferences_time.append(interval_row)

                inference_rnn_i = inference_rnn[i][0]*255
                inference_rnn_i = cv2.cvtColor(
                    inference_rnn_i, cv2.COLOR_GRAY2BGR)
                inferences_rnn.append(inference_rnn_i)
                inferences_rnn.append(interval_row)

                inference_swift_i = inference_swift[i]*255
                inference_swift_i = cv2.cvtColor(
                    inference_swift_i, cv2.COLOR_GRAY2BGR)
                inferences_swift.append(inference_swift_i)
                inferences_swift.append(interval_row)

            gts = np.concatenate(gt_list, axis=0)
            voxel_list = np.concatenate(voxel_list, axis=0)
            inferences_0 = np.concatenate(inferences_0, axis=0)
            inferences_time = np.concatenate(inferences_time, axis=0)
            inferences_rnn = np.concatenate(inferences_rnn, axis=0)
            inferences_sod = np.concatenate(inferences_sod, axis=0)
            inferences_swift = np.concatenate(inferences_swift, axis=0)

            interval_column = np.full((gts.shape[0], 1, 3), 255)

            result = np.concatenate((voxel_list, interval_column,
                                     gts, interval_column,
                                     inferences_0, interval_column,
                                     inferences_time, interval_column,
                                     inferences_rnn, interval_column,
                                     inferences_sod, interval_column,
                                     inferences_swift), axis=1)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            result_iou = str(round(our_iou.item(), 3)) + '_' + str(time_iou) + \
                '_' + str(rnn_iou) + '_' + str(sod_iou) + '_' + str(swift_iou)
            comp_score = our_iou
            result_queue.put((comp_score, result, result_iou))
            if result_queue.qsize() > n:
                result_queue.get()

    net.train()
    top_n_dir = os.path.join(save_dir, 'top_n')
    if not os.path.exists(top_n_dir):
        os.mkdir(top_n_dir)
    while not result_queue.empty():
        score, result, result_iou = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(top_n_dir, str(
            result_queue.qsize()) + '_' + result_iou + '.png')), result)
        '''
        file_dir = os.path.join(top_n_dir, str(result_queue.qsize()) + str(score))
        if not os.path.exists(file_dir):
            os.mkdir(os.path.join(file_dir))
        cv2.imwrite(os.path.join(file_dir,  'gray.png'), result[0].numpy())
        cv2.imwrite(os.path.join(file_dir,  'voxel.png'), result[1])
        cv2.imwrite(os.path.join(file_dir,  'gt.png'), result[2].numpy())
        cv2.imwrite(os.path.join(file_dir,  'inference.png'), result[3].numpy())
        '''

    return result_queue


def save_seq_predict_comp(net, dataloader, device, n, save_dir, n_frame, model_time, model_rnn, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    cnt = 0
    for images, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        cnt += 1
        batch = {}
        batch['labels'] = mask_true
        mask_true = mask_true.to(
            device=device, dtype=torch.long)  # (b n) h w -> n h w
        gts = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            Eval.reset()
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            image_copy = images.clone()
            batch['image'] = images
            mask_pred_0 = net(images)   # (b n) 1 h w
            inference_0 = mask_pred_0.cpu()
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred_0, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            our_iou = iou[1]
            if our_iou == 0:
                continue

            inference_time = model_time(images)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_time = F.one_hot(inference_time.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_time, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            time_iou = round(iou[1].item(), 3)

            inference_rnn = model_rnn(images)  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_rnn = F.one_hot(inference_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_rnn, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            rnn_iou = round(iou[1].item(), 3)

            inference_sod, _ = model_1(images)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
            Eval.reset()
            inference_swift = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_iou = round(iou[1].item(), 3)
            mask_true = mask_true.cpu()
            # gray to rgb

            voxel_list = []
            gt_list = []
            inferences_0 = []
            inferences_time = []
            inferences_rnn = []
            inferences_sod = []
            inferences_swift = []

            inference_0 = inference_0.detach().cpu().round().numpy()
            inference_sod = inference_sod.detach().cpu().round().numpy()
            inference_time = inference_time.detach().cpu().round().numpy()
            inference_rnn = inference_rnn[0, :].detach().cpu().round().numpy()
            inference_swift = inference_swift.detach().cpu().numpy()

            #interval = np.full((256, 1, 3), 255)
            interval_row = np.full((1, 336, 3), 255)

            for i in range(n_frame):
                gt = gts[i]*255
                gt = cv2.cvtColor(gt.numpy().astype(
                    np.uint8), cv2.COLOR_GRAY2BGR)
                gt_list.append(gt)
                gt_list.append(interval_row)

                voxel_image = show_voxel_grid(
                    image_copy[i].unsqueeze(0), images.shape[2], images.shape[3])
                voxel_list.append(voxel_image)
                voxel_list.append(interval_row)

                inference_0_i = inference_0[i][0]*255
                inference_0_i = cv2.cvtColor(
                    inference_0_i, cv2.COLOR_GRAY2BGR)
                inferences_0.append(inference_0_i)
                inferences_0.append(interval_row)

                inference_sod_i = inference_sod[i][0]*255
                inference_sod_i = cv2.cvtColor(
                    inference_sod_i, cv2.COLOR_GRAY2BGR)
                inferences_sod.append(inference_sod_i)
                inferences_sod.append(interval_row)

                inference_time_i = inference_time[i][0]*255
                inference_time_i = cv2.cvtColor(
                    inference_time_i, cv2.COLOR_GRAY2BGR)
                inferences_time.append(inference_time_i)
                inferences_time.append(interval_row)

                inference_rnn_i = inference_rnn[i][0]*255
                inference_rnn_i = cv2.cvtColor(
                    inference_rnn_i, cv2.COLOR_GRAY2BGR)
                inferences_rnn.append(inference_rnn_i)
                inferences_rnn.append(interval_row)

                inference_swift_i = inference_swift[i]*255
                inference_swift_i = cv2.cvtColor(
                    inference_swift_i, cv2.COLOR_GRAY2BGR)
                inferences_swift.append(inference_swift_i)
                inferences_swift.append(interval_row)

            gts = np.concatenate(gt_list, axis=0)
            voxel_list = np.concatenate(voxel_list, axis=0)
            inferences_0 = np.concatenate(inferences_0, axis=0)
            inferences_time = np.concatenate(inferences_time, axis=0)
            inferences_rnn = np.concatenate(inferences_rnn, axis=0)
            inferences_sod = np.concatenate(inferences_sod, axis=0)
            inferences_swift = np.concatenate(inferences_swift, axis=0)

            interval_column = np.full((gts.shape[0], 1, 3), 255)

            result = np.concatenate((voxel_list, interval_column,
                                     gts, interval_column,
                                     inferences_0, interval_column,
                                     inferences_time, interval_column,
                                     inferences_rnn, interval_column,
                                     inferences_sod, interval_column,
                                     inferences_swift), axis=1)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            result_iou = str(round(our_iou.item(), 3)) + '_' + str(time_iou) + \
                '_' + str(rnn_iou) + '_' + str(sod_iou) + '_' + str(swift_iou)
            cv2.imwrite(os.path.join(os.path.join(save_dir, str(
                cnt) + '_' + result_iou + '.png')), result)

    net.train()

def save_top_n_1f_predict_comp_rnn(dataloader, device, n, save_dir, n_frame, model_rnn, model_1, model_2):
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    result_queue = queue.PriorityQueue()
    cnt = 0
    for images, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        cnt += 1
        batch = {}
        batch['labels'] = mask_true
        mask_true = mask_true.to(
            device=device, dtype=torch.long)[n_frame - 1].unsqueeze(0)  # (b n) h w -> n h w
        gts = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            image = images[n_frame - 1].unsqueeze(0)  #预测最后一帧  1 c h w
            image_copy = image.clone()
            batch['image'] = image

            inference_rnn = model_rnn(images)  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_rnn = F.one_hot(inference_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_rnn[n_frame - 1].unsqueeze(0), mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            rnn_iou = iou[1]

            inference_sod, _ = model_1(image)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
            Eval.reset()
            inference_swift = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_iou = round(iou[1].item(), 3)
            mask_true = mask_true.cpu()
            # gray to rgb


            inference_sod = inference_sod.detach().cpu().round().numpy()
            inference_rnn = inference_rnn[0, n_frame-1].detach().cpu().round().numpy()
            inference_swift = inference_swift.detach().cpu().numpy()


            gt = gts[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(
                np.uint8), cv2.COLOR_GRAY2BGR)

            voxel_image = show_voxel_grid(
                image_copy, images.shape[2], images.shape[3])


            inference_rnn_i = inference_rnn[0]*255
            inference_rnn_i = cv2.cvtColor(
                inference_rnn_i, cv2.COLOR_GRAY2BGR)

            inference_sod_i = inference_sod[0][0]*255
            inference_sod_i = cv2.cvtColor(
                inference_sod_i, cv2.COLOR_GRAY2BGR)

            inference_swift_i = inference_swift[0]*255
            inference_swift_i = cv2.cvtColor(
                inference_swift_i, cv2.COLOR_GRAY2BGR)

            interval_column = np.full((gt.shape[0], 1, 3), 255)

            result = np.concatenate((voxel_image, interval_column,
                                     gt, interval_column,
                                     inference_rnn_i, interval_column,
                                     inference_sod_i, interval_column,
                                     inference_swift_i), axis=1)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            result_iou = str(round(rnn_iou.item(), 3)) + '_' + str(sod_iou) + '_' + str(swift_iou)
            comp_score = rnn_iou
            save_all_dir = os.path.join(save_dir, 'all_1f_predict')
            if not os.path.exists(save_all_dir):
                os.mkdir(save_all_dir)
            cv2.imwrite(os.path.join(os.path.join(save_all_dir, str(
                cnt) + '_' + result_iou + '.png')), result)
            #comp_score = rnn_iou
            result_queue.put((comp_score, result, result_iou))
            if result_queue.qsize() > n:
                result_queue.get()

    save_top_n_dir = os.path.join(save_dir, 'top_n_1f')
    if not os.path.exists(save_top_n_dir):
        os.mkdir(save_top_n_dir)
    while not result_queue.empty():
        score, result, result_iou = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(save_top_n_dir, str(
            result_queue.qsize()) + '_' + result_iou + '.png')), result)

def save_top_n_1f_predict_comp(net, dataloader, device, n, save_dir, n_frame, model_rnn, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    result_queue = queue.PriorityQueue()
    cnt = 0
    for images, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        cnt += 1
        batch = {}
        batch['labels'] = mask_true
        mask_true = mask_true.to(
            device=device, dtype=torch.long)[n_frame - 1].unsqueeze(0)  # (b n) h w -> n h w
        gts = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            Eval.reset()
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            image = images[n_frame - 1].unsqueeze(0)  #预测最后一帧  1 c h w
            image_copy = image.clone()
            batch['image'] = image
            mask_pred_0 = net(image)   # (b n) 1 h w
            inference_0 = mask_pred_0.cpu()
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred_0, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            our_iou = iou[1]

            inference_rnn = model_rnn(images)  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_rnn = F.one_hot(inference_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_rnn[n_frame - 1].unsqueeze(0), mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            rnn_iou = iou[1]

            inference_sod, _ = model_1(image)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
            Eval.reset()
            inference_swift = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_iou = round(iou[1].item(), 3)
            mask_true = mask_true.cpu()
            # gray to rgb


            inference_0 = inference_0.detach().cpu().round().numpy()
            inference_sod = inference_sod.detach().cpu().round().numpy()
            inference_rnn = inference_rnn[0, n_frame-1].detach().cpu().round().numpy()
            inference_swift = inference_swift.detach().cpu().numpy()


            gt = gts[0]*255
            gt = cv2.cvtColor(gt.numpy().astype(
                np.uint8), cv2.COLOR_GRAY2BGR)

            voxel_image = show_voxel_grid(
                image_copy, images.shape[2], images.shape[3])

            inference_0_i = inference_0[0][0]*255
            inference_0_i = cv2.cvtColor(
                inference_0_i, cv2.COLOR_GRAY2BGR)

            inference_rnn_i = inference_rnn[0]*255
            inference_rnn_i = cv2.cvtColor(
                inference_rnn_i, cv2.COLOR_GRAY2BGR)

            inference_sod_i = inference_sod[0][0]*255
            inference_sod_i = cv2.cvtColor(
                inference_sod_i, cv2.COLOR_GRAY2BGR)



            inference_swift_i = inference_swift[0]*255
            inference_swift_i = cv2.cvtColor(
                inference_swift_i, cv2.COLOR_GRAY2BGR)

            interval_column = np.full((gt.shape[0], 1, 3), 255)

            result = np.concatenate((voxel_image, interval_column,
                                     gt, interval_column,
                                     inference_0_i, interval_column,
                                     inference_rnn_i, interval_column,
                                     inference_sod_i, interval_column,
                                     inference_swift_i), axis=1)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            result_iou = str(round(our_iou.item(), 3)) + \
                '_' + str(round(rnn_iou.item(), 3)) + '_' + str(sod_iou) + '_' + str(swift_iou)
            comp_score = rnn_iou
            save_all_dir = os.path.join(save_dir, 'all_1f_predict')
            if not os.path.exists(save_all_dir):
                os.mkdir(save_all_dir)
            cv2.imwrite(os.path.join(os.path.join(save_all_dir, str(
                cnt) + '_' + result_iou + '.png')), result)
            #comp_score = rnn_iou
            result_queue.put((comp_score, result, result_iou))
            if result_queue.qsize() > n:
                result_queue.get()

    net.train()
    save_top_n_dir = os.path.join(save_dir, 'top_n_1f')
    if not os.path.exists(save_top_n_dir):
        os.mkdir(save_top_n_dir)
    while not result_queue.empty():
        score, result, result_iou = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(save_top_n_dir, str(
            result_queue.qsize()) + '_' + result_iou + '.png')), result)

def save_top_n_seq_predict_comp(net, dataloader, device, n, save_dir, n_frame, model_time, model_rnn, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    Eval = iouEval(nClasses=2)
    # iterate over the validation set
    result_queue = queue.PriorityQueue()
    cnt = 0
    for images, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        cnt += 1
        batch = {}
        batch['labels'] = mask_true
        mask_true = mask_true.to(
            device=device, dtype=torch.long)  # (b n) h w -> n h w
        gts = mask_true.cpu()
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            Eval.reset()
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            image_copy = images.clone()
            batch['image'] = images
            mask_pred_0 = net(images)   # (b n) 1 h w
            inference_0 = mask_pred_0.cpu()
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            Eval.addBatch(mask_pred_0, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            our_iou = iou[1]
            if our_iou == 0:
                continue

            inference_time = model_time(images)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_time = F.one_hot(inference_time.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_time, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            time_iou = round(iou[1].item(), 3)

            inference_rnn = model_rnn(images)  # b n 1 h w -> 1 n 1 h w
            Eval.reset()
            mask_rnn = F.one_hot(inference_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_rnn, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            rnn_iou = round(iou[1].item(), 3)

            inference_sod, _ = model_1(images)  # (b n ) 1 h w -> n 1 h w
            Eval.reset()
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_sod, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            sod_iou = round(iou[1].item(), 3)

            logits, _ = model_2.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
            Eval.reset()
            inference_swift = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            Eval.addBatch(mask_swift, mask_true)
            iou_mean, iou, acc_mean, acc = Eval.getIoU()
            swift_iou = round(iou[1].item(), 3)
            mask_true = mask_true.cpu()
            # gray to rgb

            voxel_list = []
            gt_list = []
            inferences_0 = []
            inferences_time = []
            inferences_rnn = []
            inferences_sod = []
            inferences_swift = []

            inference_0 = inference_0.detach().cpu().round().numpy()
            inference_sod = inference_sod.detach().cpu().round().numpy()
            inference_time = inference_time.detach().cpu().round().numpy()
            inference_rnn = inference_rnn[0, :].detach().cpu().round().numpy()
            inference_swift = inference_swift.detach().cpu().numpy()

            #interval = np.full((256, 1, 3), 255)
            interval_row = np.full((1, 336, 3), 255)

            for i in range(n_frame):
                gt = gts[i]*255
                gt = cv2.cvtColor(gt.numpy().astype(
                    np.uint8), cv2.COLOR_GRAY2BGR)
                gt_list.append(gt)
                gt_list.append(interval_row)

                voxel_image = show_voxel_grid(
                    image_copy[i].unsqueeze(0), images.shape[2], images.shape[3])
                voxel_list.append(voxel_image)
                voxel_list.append(interval_row)

                inference_0_i = inference_0[i][0]*255
                inference_0_i = cv2.cvtColor(
                    inference_0_i, cv2.COLOR_GRAY2BGR)
                inferences_0.append(inference_0_i)
                inferences_0.append(interval_row)

                inference_sod_i = inference_sod[i][0]*255
                inference_sod_i = cv2.cvtColor(
                    inference_sod_i, cv2.COLOR_GRAY2BGR)
                inferences_sod.append(inference_sod_i)
                inferences_sod.append(interval_row)

                inference_time_i = inference_time[i][0]*255
                inference_time_i = cv2.cvtColor(
                    inference_time_i, cv2.COLOR_GRAY2BGR)
                inferences_time.append(inference_time_i)
                inferences_time.append(interval_row)

                inference_rnn_i = inference_rnn[i][0]*255
                inference_rnn_i = cv2.cvtColor(
                    inference_rnn_i, cv2.COLOR_GRAY2BGR)
                inferences_rnn.append(inference_rnn_i)
                inferences_rnn.append(interval_row)

                inference_swift_i = inference_swift[i]*255
                inference_swift_i = cv2.cvtColor(
                    inference_swift_i, cv2.COLOR_GRAY2BGR)
                inferences_swift.append(inference_swift_i)
                inferences_swift.append(interval_row)

            gts = np.concatenate(gt_list, axis=0)
            voxel_list = np.concatenate(voxel_list, axis=0)
            inferences_0 = np.concatenate(inferences_0, axis=0)
            inferences_time = np.concatenate(inferences_time, axis=0)
            inferences_rnn = np.concatenate(inferences_rnn, axis=0)
            inferences_sod = np.concatenate(inferences_sod, axis=0)
            inferences_swift = np.concatenate(inferences_swift, axis=0)

            interval_column = np.full((gts.shape[0], 1, 3), 255)

            result = np.concatenate((voxel_list, interval_column,
                                     gts, interval_column,
                                     inferences_0, interval_column,
                                     inferences_time, interval_column,
                                     inferences_rnn, interval_column,
                                     inferences_sod, interval_column,
                                     inferences_swift), axis=1)
            '''
            result = [gray_image[0], voxel_image, gt[0] * 255,
            inference[0][0].round() * 255]
            '''
            result_iou = str(round(our_iou.item(), 3)) + '_' + str(time_iou) + \
                '_' + str(rnn_iou) + '_' + str(sod_iou) + '_' + str(swift_iou)
            comp_score = our_iou
            save_all_dir = os.path.join(save_dir, 'all_seq_predict')
            if not os.path.exists(save_all_dir):
                os.mkdir(save_all_dir)
            cv2.imwrite(os.path.join(os.path.join(save_all_dir, str(
                cnt) + '_' + result_iou + '.png')), result)
            #comp_score = rnn_iou
            result_queue.put((comp_score, result, result_iou))
            if result_queue.qsize() > n:
                result_queue.get()

    net.train()
    save_top_n_dir = os.path.join(save_dir, 'top_n_seq')
    if not os.path.exists(save_top_n_dir):
        os.mkdir(save_top_n_dir)
    while not result_queue.empty():
        score, result, result_iou = result_queue.get()
        cv2.imwrite(os.path.join(os.path.join(save_top_n_dir, str(
            result_queue.qsize()) + '_' + result_iou + '.png')), result)


def eval_iou_deck(net, dataloader, device, save_dir, model_time, model_rnn, model_1, model_2):
    net.eval()
    num_val_batches = len(dataloader)
    eval_model = ('unet_cs', 'unet_time', 'RMS-net', 'sodmodel', 'swift')
    eval_dict = {}
    for model_name in eval_model:
        eval_dict[model_name] = iouEval(nClasses=2)
        eval_dict[model_name].reset()

    # iterate over the validation set
    cnt = 0
    for images, mask_true, _ in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # move images and labels to correct device and type
        cnt += 1
        batch = {}
        batch['labels'] = mask_true
        mask_true = mask_true.to(
            device=device, dtype=torch.long)  # (b n) h w -> n h w
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            image_copy = images.clone()
            batch['image'] = images
            mask_pred_0 = net(images)   # (b n) 1 h w
            inference_0 = mask_pred_0.cpu()
            mask_pred_0 = F.one_hot(mask_pred_0.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            eval_dict[eval_model[0]].addBatch(mask_pred_0, mask_true)

            inference_time = model_time(images)  # (b n ) 1 h w -> n 1 h w
            mask_time = F.one_hot(inference_time.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            eval_dict[eval_model[1]].addBatch(mask_time, mask_true)

            inference_rnn = model_rnn(images)  # b n 1 h w -> 1 n 1 h w
            mask_rnn = F.one_hot(inference_rnn.round()[
                                 0].squeeze(1).long(), 2).permute(0, 3, 1, 2).float()
            eval_dict[eval_model[2]].addBatch(mask_rnn, mask_true)

            inference_sod, _ = model_1(images)  # (b n ) 1 h w -> n 1 h w
            mask_sod = F.one_hot(inference_sod.round().squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            eval_dict[eval_model[3]].addBatch(mask_sod, mask_true)

            logits, _ = model_2.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
            inference_swift = torch.argmax(logits.data, dim=1).float()
            mask_swift = F.one_hot(inference_swift.squeeze(
                1).long(), 2).permute(0, 3, 1, 2).float()
            eval_dict[eval_model[4]].addBatch(mask_swift, mask_true)

    iou_mean, iou, _, _ = eval_dict[eval_model[0]].getIoU()
    our_iou = iou[1]
    our_miou = round(iou_mean.item(), 3)
    iou_mean, iou, _, _ = eval_dict[eval_model[1]].getIoU()
    time_iou = round(iou[1].item(), 3)
    time_miou = round(iou_mean.item(), 3)
    iou_mean, iou, _, _ = eval_dict[eval_model[2]].getIoU()
    rnn_iou = round(iou[1].item(), 3)
    rnn_miou = round(iou_mean.item(), 3)
    iou_mean, iou, _, _ = eval_dict[eval_model[3]].getIoU()
    sod_iou = round(iou[1].item(), 3)
    sod_miou = round(iou_mean.item(), 3)
    iou_mean, iou, _, _ = eval_dict[eval_model[4]].getIoU()
    swift_iou = round(iou[1].item(), 3)
    swift_moiu = round(iou_mean.item(), 3)

    result_txt = f'{eval_model[0]} : {our_iou}, {our_miou}\n' + f'{eval_model[1]} : {time_iou}, {time_miou}\n' + \
        f'{eval_model[2]} : {rnn_iou}, {rnn_miou}\n' + f'{eval_model[3]} : {sod_iou}, {sod_miou}\n' + \
        f'{eval_model[4]} : {swift_iou}, {swift_moiu}\n'
    '''
    result_txt = eval_model[0] + ': ' + str(our_iou) + '  ' + str(our_miou) + '\n' 
    + eval_model[1] + ': ' + str(time_iou) + '  ' + str(time_miou) + '\n'
    + eval_model[2] + ': ' + str(rnn_iou) + '  ' + str(rnn_miou) + '\n'
    + eval_model[3] + ': ' + str(sod_iou) + '  ' + str(sod_miou) + '\n'
    + eval_model[4] + ': ' + str(swift_iou) + '  ' + str(swift_moiu) + '\n'
    '''
    with open(os.path.join(save_dir, 'log.txt'), 'w') as f:
        f.write(result_txt)
    f.close()
    print(result_txt)
    net.train()
