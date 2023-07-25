import torch
from torch import Tensor, nn
import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def depth_metric_reconstruction_loss(depth, target, weights=None, loss='L1', normalize=False):
    def one_scale(depth, target, loss_function, normalize):
        b, h, w = depth.size()

        target_scaled = F.interpolate(target.unsqueeze(1), size=(h, w), mode='area')[:,0]

        diff = depth-target_scaled

        if normalize:
            diff = diff/target_scaled

        return loss_function(diff, depth.detach()*0)

    if weights is not None:
        assert(len(weights) == len(depth))
    else:
        weights = [1 for d in depth]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    if type(loss) is str:
        assert(loss in ['L1', 'MSE', 'SmoothL1'])

        if loss == 'L1':
            loss_function = torch.nn.L1Loss()
        elif loss == 'MSE':
            loss_function = torch.nn.MSELoss()
        elif loss == 'SmoothL1':
            loss_function = torch.nn.SmoothL1Loss()
    else:
        loss_function = loss

    loss_output = 0
    for d, w in zip(depth, weights):
        loss_output += w*one_scale(d, target, loss_function, normalize)
    return loss_output


if __name__ == '__main__':
    input = torch.randn((4, 1, 260, 346), requires_grad=True) % 1
    gt = torch.randn((4, 260, 346)) % 1
    loss_1 = dice_loss(input=input, target=gt.unsqueeze(1))
    print(loss_1)
    loss_2 = F.binary_cross_entropy(input,gt.unsqueeze(1))
    print(loss_2)
    LOSS = FocalLoss(logits=True)
    loss_3 = LOSS(input, gt.unsqueeze(1))
    print(loss_3)

    loss_combine = loss_1 + loss_3