import time
import torch

def evaluate_sod_runtime(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    # iterate over the validation set
    run_time = 0
    for image, mask_true, _, _ in dataloader:
        # move images and labels to correct device and type

        start_time = time.time()
        with torch.no_grad():
            image = image.to(device=device, dtype=torch.float32)
            #mask_pred = net(image)
            mask_pred, _ = net(image)
            #mask_pred = F.one_hot(mask_pred.round().squeeze(
            #    1).long(), 2).permute(0, 3, 1, 2).float()
        end_time = time.time()
        runtime_each_frame = (end_time - start_time) 
        run_time += runtime_each_frame
    return run_time / num_val_batches

def eval_swift_runtime(net, dataloader, device):
    num_val_batches = len(dataloader)
    # iterate over the validation set
    run_time = 0
    for images, mask_true, _, _ in dataloader:
        # move images and labels to correct device and type
        start_time = time.time()
        batch = {}
        batch['labels'] = mask_true

        with torch.no_grad():
            # predict the mask
            # (b n) c h w -> n c h w
            images = images.to(device=device, dtype=torch.float32)
            batch['image'] = images

            logits, _ = net.do_forward(
                batch, batch['labels'].shape[1:3])  # n 2 h w
        end_time = time.time()
        runtime_each_frame = (end_time - start_time)
        run_time += runtime_each_frame
    return run_time / num_val_batches


def evaluate_recurrentUnet_runtime(net, dataloader, device, frame_deck=False, n_frame=4, key_idx=0):
    net.eval()
    num_val_batches = len(dataloader)
    #print(num_val_batches)
    # iterate over the validation set
    run_time = 0
    for image, mask_true, _, _ in dataloader:
        # move images and labels to correct device and type
        #mask_true = mask_true[:, key_idx, :]
        start_time = time.time()
        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)  # b n 1 h w
        #    mask_pred = mask_pred[:, n_frame - 1]  # b 1 h w
        end_time = time.time()
        runtime_each_frame = (end_time - start_time) / n_frame
        run_time += runtime_each_frame
    return run_time / num_val_batches
