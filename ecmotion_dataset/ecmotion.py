from dis import dis
from email.mime import image
import functools
import glob
import logging
from operator import index
from tkinter.messagebox import NO
from tkinter.ttk import LabelFrame
import numpy as np
import os
import cv2
import torch
import torchvision.transforms as F

from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union, Tuple
from .multiprocessing import TaskManager
from .event_dm import EventDataModule
from .Voxel import VoxelGrid

def undistort_img(img, K, D):
    if (K is None):
        return img
    if (D is None):
        D = np.array([0, 0, 0, 0])

    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted

def mask_to_color(mask):
    colors = [[255,255,0], [0,255,255], [255,0,255],
                [0,255,0],   [0,0,255],   [255,0,0]]

    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    m_ = np.max(mask) + 500
    m_ = max(m_, 3500)
    i = 0
    while (m_ > 0):
        cmb[mask < m_] = np.array(colors[i % len(colors)])
        i += 1
        m_ -= 1000

    cmb[mask < 500] = np.array([0,0,0])
    return cmb

def mask_to_color_anchor(mask):
    colors = [[255,255,0], [0,255,255], [255,0,255],
                [0,255,0],   [0,0,255],   [255,0,0]]

    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    anchor_boxes = []
    m_ = np.max(mask) + 500
    m_ = max(m_, 3500)
    i = 0
    while (m_ > 0):
        mask_ = (mask < m_) & (mask > m_ - 1000) & (mask > 500)
        cmb[mask_] = np.array(colors[i % len(colors)])
        print(mask_)
        i += 1
        m_ -= 1000
        z = np.bincount(mask_[(mask_ >= 0) & (mask_ < 2)])
        print(len(z))
        if len(z) == 1 or z[1] / np.sum(z) < 0.001:
            continue
        mask_pos = np.where(mask_)
        x_min = mask_pos[0].min()
        x_max = mask_pos[0].max()
        y_min = mask_pos[1].min()
        y_max = mask_pos[1].max()
        #anchor_boxes.append([x_min + (x_max - x_min)/2, y_min + (y_max - y_min)/2, x_max - x_max, y_max - y_min])
        anchor_boxes.append([x_min, y_max, x_max - x_min, y_max - y_min, 1])

    cmb[mask < 500] = np.array([0,0,0])
    return cmb, anchor_boxes

def accumulate_image(data, imageH=260, imageW=346, noise_show=False):
    """
    Parameters
    ----------
    data : [3, num_sample]: [x, y, t] or [4, num_sample]: [x, y, t, p]
    Returns
    -------
    image : accumulate image

    """
    x = data[0, :]  # x
    y = data[1, :]  # y
    y = imageH - y - 1
    t = data[2, :]  # t
    if data.shape[0] == 4:
        p = data[3, :]  # p

    img_cam = np.zeros([imageH, imageW])
    num_sample = len(x)

    for idx in range(num_sample):
        # coordx = int(x[idx]) - 1
        # coordy = 260 - int(y[idx]) - 1

        coordx = int(x[idx])  # [0, 345]
        coordy = imageH - int(y[idx]) - 1  # 镜像同时仍然变为[0, 259]

        img_cam[coordy, coordx] = img_cam[coordy, coordx] + 1

    if noise_show:
        img_cam *= 255.0
        image_cam_accumulate = img_cam.astype(np.uint8)
    else:
        image_cam_accumulate = normalizeImage3Sigma(img_cam, imageH, imageW)
        image_cam_accumulate = image_cam_accumulate.astype(np.uint8)

    return image_cam_accumulate


def normalizeImage3Sigma(image, imageH=260, imageW=346):
    """followed by matlab dhp19 generate"""
    sum_img = np.sum(image)
    count_image = np.sum(image > 0)
    mean_image = sum_img / count_image
    var_img = np.var(image[image > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    numSDevs = 3.0
    # Rectify polarity=true
    meanGrey = 0
    range_old = numSDevs * sig_img
    half_range = 0
    range_new = 255
    # Rectify polarity=false
    # meanGrey=127 / 255
    # range= 2*numSDevs * sig_img
    # halfrange = numSDevs * sig_img

    normalizedMat = np.zeros([imageH, imageW])
    for i in range(imageH):
        for j in range(imageW):
            l = image[i, j]
            if l == 0:
                normalizedMat[i, j] = meanGrey
            else:
                f = (l + half_range) * range_new / range_old
                if f > range_new:
                    f = range_new

                if f < 0:
                    f = 0
                normalizedMat[i, j] = np.floor(f)

    return normalizedMat

def get_slice(cloud, idx, ts, width, mode=0, idx_step=0.01):
    if (cloud.shape[0] == 0):
        return cloud, np.array([0])

    ts_lo = ts
    ts_hi = ts + width
    if (mode == 1):
        ts_lo = ts - width / 2.0
        ts_hi = ts + width / 2.0
    if (mode == 2):
        ts_lo = ts - width
        ts_hi = ts
    if (ts_lo < 0): ts_lo = 0

    t0 = cloud[0][0]

    idx_lo = int((ts_lo - t0) / idx_step)
    idx_hi = int((ts_hi - t0) / idx_step)
    if (idx_lo >= len(idx)): idx_lo = -1
    if (idx_hi >= len(idx)): idx_hi = -1

    sl = np.copy(cloud[idx[idx_lo]:idx[idx_hi]].astype(np.float64))
    idx_ = np.copy(idx[idx_lo:idx_hi])

    if (idx_lo == idx_hi):
        return sl, np.array([0])

    if (len(idx_) > 0):
        idx_0 = idx_[0]
        idx_ -= idx_0

    if (sl.shape[0] > 0):
        t0 = sl[0][0]
        sl[:,0] -= t0
    
    return sl, idx_

def PositiveLabelOfSum(label):
    cnt = np.zeros(2)
    #label = label.detach().cpu().numpy()
    mask = (label >= 0) & (label < 2)
    Label = label[mask].astype(np.uint8)
    cnt += np.bincount(Label, minlength=2)
    res = (cnt[1]) / (np.sum(cnt))
    return res

def resize_img(image, mask=None, target_size=None):
    trans = F.Resize(target_size)
    image = trans(image)
    if mask is not None:
        mask = trans(mask)
    return image, mask if mask is not None else image
    
def pad_resize_image(inp_img, out_img=None, out_img_1=None, target_size=None):
    """
    Function to pad and resize images to a given size.
    out_img is None only during inference. During training and testing
    out_img is NOT None.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image of mask.
    :param target_size: The size of the final images.
    :return: Re-sized inp_img and out_img
    """
    h, w, c = inp_img.shape
    size = max(h, w)

    padding_h = (size - h) // 2
    padding_w = (size - w) // 2

    if out_img is None:
        # For inference
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x
    else:
        # For training and testing
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        temp_y = cv2.copyMakeBorder(out_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        temp_z = cv2.copyMakeBorder(out_img_1, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # print(inp_img.shape, temp_x.shape, out_img.shape, temp_y.shape)

        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
            temp_y = cv2.resize(temp_y, (target_size, target_size), interpolation=cv2.INTER_AREA)
            temp_z = cv2.resize(temp_z, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x, temp_y, temp_z

class ecmotion(EventDataModule):

    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8,
                 pin_memory: bool = False, resize: bool = False, frame_deck: bool = False, n_frame: int = 2):
        super(ecmotion, self).__init__(img_shape=(260, 346), batch_size=batch_size, shuffle=shuffle,
                                          num_workers=num_workers, pin_memory=pin_memory, resize=resize, 
                                          frame_deck=frame_deck, n_frame=n_frame)
        pre_processing_params = {"r": 3.0, "d_max": 32, "n_samples": 4096*2, "sampling": True}
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    @staticmethod
    def read_label(raw_file: str) -> Optional[Union[str, List[str]]]:
        return raw_file.split("/")[-2]
    
    @staticmethod
    def load(raw_file: str, interval: int):
        DataList_3_channel = []
        MaskList = []
        gray_list = []
        path, _ = os.path.splitext(raw_file)
        image_path = path.replace("/media/shaobo/shaobo/dataSet/evimo", "/media/shaobo/shaobo/dataSet/evimo/txt").replace("npz", "txt")
        image_path = os.path.join(image_path, "img")
        sl_npz = np.load(raw_file, allow_pickle=True)
        raw_data = sl_npz['events']
        idx = sl_npz['index']
        meta = sl_npz['meta'].item()
        image_file = []
        gt_ts = []
        for frame in meta['frames']:
            gt_ts.append(frame['ts'])
            image_file.append(frame['classical_frame'])
        
        gt_ts = np.array(gt_ts)
        discretization = sl_npz['discretization']
        slice_width = 0.03
        first_ts = raw_data[0][0]
        last_ts = raw_data[-1][0]
        K = sl_npz['K']
        D = sl_npz['D']
        meta = sl_npz['meta'].item()
        mask_gt = sl_npz['mask']
        voxel_3_channel = VoxelGrid((3, 260, 346), normalize=True)
        #voxel_1_channel = VoxelGrid((1, 260, 346), normalize=True)
        for i, ts in enumerate(gt_ts):
            if (ts - slice_width / 2  < first_ts or ts + slice_width / 2 > last_ts):
                continue
            if i % interval == 0:
            #if i % interval == 0:
                sl, _ = get_slice(raw_data, idx, ts, slice_width, mode=1, idx_step=discretization)
                all_x = sl[:,1]
                all_y = sl[:,2]
                all_p = sl[:,3]
                all_ts = sl[:,0]
                #all_p = all_p.astype(np.float64)
                all_p[all_p == 0] = -1
                sl = np.column_stack((all_x, all_y, all_ts, all_p))
                sl = torch.from_numpy(sl)
                voxel_image_3_channel = voxel_3_channel.convert(sl)
                image_file_path = os.path.join(image_path, image_file[i])
                gray_image = cv2.imread(image_file_path, 0)
                gray_list.append(gray_image[2:258, 5:341])
                #voxel_image_1_channel = voxel_1_channel.convert(sl)
                #sl = torch.from_numpy(sl).float()
                DataList_3_channel.append(voxel_image_3_channel[:, 2:258, 5:341])
                #DataList_1_channel.append(voxel_image_1_channel)
                mask = undistort_img(mask_gt[i], K, D)
                col_mask = mask_to_color(mask)
                gray_mask = cv2.cvtColor(col_mask, cv2.COLOR_BGR2GRAY)
                binary_mask = cv2.threshold(gray_mask, 30, 255, cv2.THRESH_BINARY)[1]
                #binary_mask = cv2.adaptiveThreshold(gray_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY)[1]
                cv2.normalize(binary_mask, binary_mask, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
                #binary_mask = torch.from_numpy(binary_mask)
                MaskList.append(binary_mask[2:258, 5:341])
        return DataList_3_channel, MaskList, gray_list

    @functools.lru_cache(maxsize=100)
    def map_label(self, label: str) -> int:
        label_dict = {lbl: i for i, lbl in enumerate(self.classes)}
        return label_dict.get(label, None)

    def _load_processed_file(self, f_path: str):
        return torch.load(f_path)

    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    def _prepare_dataset(self, mode: str):
        processed_dir = os.path.join(self.root, "processed_framedeck")
        raw_files = self.raw_files(mode)
        #class_dict = {class_id: i for i, class_id in enumerate(self.classes)}
        kwargs = dict(load_func=self.load, interval=2 if mode == "training" else 1, read_annotations=self.read_annotations, resize=self.resize,
                      frame_deck=self.frame_deck, n_frame=self.n_frame)
        logging.debug(f"Found {len(raw_files)} raw files in dataset (mode = {mode})")

        task_manager = TaskManager(self.num_workers, queue_size=self.num_workers)
        for rf in tqdm(raw_files):
            processed_file = rf.replace(self.root, processed_dir)
            task_manager.queue(self.processing, rf=rf, pf=processed_file, **kwargs)
        task_manager.join()

    @staticmethod
    def processing(rf: str, pf: str, interval: int, resize: bool, frame_deck: bool, n_frame: int, load_func: Callable,
                   read_annotations: Callable[[str], np.ndarray]):
        rf_wo_ext, _ = os.path.splitext(rf)

        # Load data from raw file. If the according loaders are available, add annotation, label and class id.
        data_list, mask_list, gray_list = load_func(rf, interval)
        #assert len(data_list) == len(label_dict), "error: data_list and label_dict have different lengths"
        if frame_deck is True:
            for i, data in enumerate(data_list):
                if i + n_frame < len(data_list):
                    obj_dict = dict()
                    '''
                    training = ['box', 'floor', 'table', 'tabletop', 'tabletop-egomotion', 'wall']
                    validation = ['box', 'fast', 'floor', 'table', 'tabletop', 'wall']
                    '''
                    file_class = rf.split("/")[-3]
                    image_deck = [data_list[i+j].unsqueeze(0) for j in range(n_frame)]
                    mask_deck = [torch.from_numpy(mask_list[i+j]).unsqueeze(0) for j in range(n_frame)]
                    gray_deck = [torch.from_numpy(gray_list[i+j]).unsqueeze(0) for j in range(n_frame)]
                    images = torch.cat(image_deck, 0)
                    masks = torch.cat(mask_deck, 0)
                    grays = torch.cat(gray_deck, 0)
                    flag = 1
                    for label in mask_deck:
                        if PositiveLabelOfSum(label.numpy()) < 0.03:
                            flag = 0
                            break
                    if flag == 0:
                        continue
                    obj_dict['class'] = file_class
                    obj_dict['voxel_image'] = images
                    obj_dict['mask'] = masks
                    obj_dict['gray'] = grays
                    os.makedirs(os.path.dirname(pf), exist_ok=True)
                    torch.save(obj_dict, pf.replace(".npz",'')+str(i).rjust(5, "0"))
        else:
            for i, data in enumerate(data_list):
                #if i % interval == 0:
                    obj_dict = dict()
                    '''
                    training_class = ['box', 'floor', 'table', 'tabletop', 'tabletop-egomotion', 'wall']
                    validation_class = ['box', 'fast', 'floor', 'table', 'tabletop', 'wall']
                    '''
                    file_class = rf.split("/")[-3]
                    mask = mask_list[i]
                    if PositiveLabelOfSum(mask) < 0.03:
                        continue
                    mask = torch.from_numpy(mask)
                    #bounding_boxes = torch.tensor(bounding_box_list[i])
                    obj_dict['class'] = file_class
                    obj_dict['voxel_image'] = data
                    #obj_dict['voxel_image_1_channel'] = data_list_1_channel[i]
                    obj_dict['mask'] = mask
                    #obj_dict['bounding_box'] = bounding_boxes
                    
                    os.makedirs(os.path.dirname(pf), exist_ok=True)
                    torch.save(obj_dict, pf.replace(".npz",'')+str(i).rjust(5, "0"))
                    
    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    def raw_files(self, mode: str) -> List[str]:
        return glob.glob(os.path.join(self.root, mode, "*", "npz", "*.npz"), recursive=True)

    def processed_files(self, mode: str) -> List[str]:
        processed_dir = os.path.join(self.root, "processed")
        return glob.glob(os.path.join(processed_dir, mode, "*"))

    @property
    def classes(self) -> List[str]:
        #return os.listdir(os.path.join(self.root, "raw"))
        return None

if __name__ == "__main__":
    slices=[[[]]]
    a = [1, 2, 3] ; b = [4, 5, 6]; c = [7, 8, 9]
    print(glob.glob(os.path.join("/media/shaobo/shaobo/dataSet/EV-IMO/training", "*", "npz", "*.npz"), recursive=True))

    # 多个一维数组，一行一行堆叠形成一个二维数组（3x3）
    # 两个二维数组，构成三维数组（2x3x3）
    # 最后再用一个 [] 把整体括起来！
    w2 = np.array( [ [a,b,c], [a,b,c] ] )
    print(w2)