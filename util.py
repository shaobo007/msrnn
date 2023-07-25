import os
import torch
import numpy as np
from tqdm import tqdm

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def calculate_weight_labels(dataloader, num_classes):
    z = np.zeros((num_classes))
    tqdm_batch = tqdm(dataloader)
    print("Calculating classes weights")
    for sample in tqdm_batch:
        y = sample.y
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join('./', 'class_weights.npy')
    np.save(classes_weights_path, ret)
    return ret


class iouEval:

    def __init__(self, nClasses, ignoreIndex=20):

        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1  # if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset(self):
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()
        self.cdp_obstacle = torch.zeros(1).double()
        self.tp_obstacle = torch.zeros(1).double()
        self.idp_obstacle = torch.zeros(1).double()
        self.tp_nonobstacle = torch.zeros(1).double()
        # self.cdi = torch.zeros(1).double()

    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be "batch_size x nClasses x H x W"
        # cdi = 0

        # print ("X is cuda: ", x.is_cuda)
        # print ("Y is cuda: ", y.is_cuda)
        '''
        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()
        '''
        device = x.device
        x = x.to(dtype=torch.int64)
        y = y.to(dtype=torch.int64)
        # if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.to(device)
            x_onehot.scatter_(1, x, 1).float()  # dim index src 按照列用1替换0，索引为x
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.to(device)
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1):
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)  # 加一维
            x_onehot = x_onehot[:, :self.ignoreIndex]  # ignoreIndex后的都不要
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0
            
        tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fpmult = x_onehot * (
                    1 - y_onehot - ignores)  # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()
        fnmult = (1 - x_onehot) * (y_onehot)  # times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                       keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()
        '''
        cdp_obstacle = tpmult[:, 19].sum()  # obstacle index 19
        tp_obstacle = y_onehot[:, 19].sum()

        idp_obstacle = (x_onehot[:, 19] - tpmult[:, 19]).sum()
        tp_nonobstacle = (-1*y_onehot+1).sum()

        # for i in range(0, x.size(0)):
        #     if tpmult[i].sum()/(y_onehot[i].sum() + 1e-15) >= 0.5:
        #         cdi += 1


        self.cdp_obstacle += cdp_obstacle.double().cpu()
        self.tp_obstacle += tp_obstacle.double().cpu()
        self.idp_obstacle += idp_obstacle.double().cpu()
        self.tp_nonobstacle += tp_nonobstacle.double().cpu()
        # self.cdi += cdi.double().cpu()
        '''

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        iou_not_zero = list(filter(lambda x: x != 0, iou))
        # print(len(iou_not_zero))
        iou_mean = sum(iou_not_zero) / len(iou_not_zero)
        tfp = self.tp + self.fp + 1e-15
        acc = num / tfp
        acc_not_zero = list(filter(lambda x: x != 0, acc))
        acc_mean = sum(acc_not_zero) / len(acc_not_zero)

        return iou_mean, iou, acc_mean, acc  # returns "iou mean", "iou per class"
    '''
    def getObstacleEval(self):

        pdr_obstacle = self.cdp_obstacle / (self.tp_obstacle+1e-15)
        pfp_obstacle = self.idp_obstacle / (self.tp_nonobstacle+1e-15)

        return pdr_obstacle, pfp_obstacle
    '''
    
def calculate_iou(pred, y, dim=1):
    pred = torch.argmax(pred, dim=dim)
    tpmult = pred * y
    fpmult = pred * (1 - y)
    fnmult = (1 - pred) * y
    tp = torch.sum(tpmult).double().cpu()
    fp = torch.sum(fpmult).double().cpu()
    fn = torch.sum(fnmult).double().cpu()
    iou = tp / (tp + fp + fn)
    return iou

def eval_iou(model, device, data_loader):
    """Evaluate the model on the dataset."""
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for sample in tqdm(data_loader):
            data = sample.to(device)
            end_data = model(data)
            x = end_data.x
            y = end_data.y
            iou = calculate_iou(x, y)
            metric.add(iou, 1)
    return metric[0]/metric[1]

def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float((cmp.type(y.dtype)).sum()) / len(y)


def evaluate_acc(model, device, data_loader):
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for sample in tqdm(data_loader):
            data = sample.to(device)
            end_data = model(data)
            x = end_data.x
            acc = accuracy(x, data.y)
            metric.add(acc, 1)

    return metric[0]/metric[1]

if __name__ == '__main__':
    Eval = iouEval(2)
    x1 = torch.tensor(([0, 0, 0, 0, 0],[0, 0, 0, 0, 0])).unsqueeze(0)
    x1 = torch.cat([x1,x1], 0).unsqueeze(1)
    y = torch.tensor(([0, 0, 0, 0, 0],[0, 0, 0, 0, 0])).unsqueeze(0)
    y = torch.cat([y,y], 0).unsqueeze(1)
    print(y.shape)
    Eval.addBatch(x1,y)
    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    print(iou_mean)
    print(iou)
    print(acc)
    x1 = torch.tensor(([0, 1, 0, 0, 0],[0, 0, 0, 0, 0])).unsqueeze(0)
    x1 = torch.cat([x1,x1], 0).unsqueeze(1)
    y = torch.tensor(([0, 1, 0, 0, 0],[0, 0, 0, 0, 0])).unsqueeze(0)
    y = torch.cat([y,y], 0).unsqueeze(1)
    Eval.addBatch(x1,y)
    iou_mean, iou, acc_mean, acc = Eval.getIoU()
    Eval.addBatch(x1,y)
    print(iou_mean)
    print(iou)
    print(acc)