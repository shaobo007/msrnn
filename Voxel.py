import numpy as np
import cv2
import torch
from enum import Enum, auto

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()


class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events[:, 1].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events[:, 2]
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            x0 = events[:, 0].int()
            y0 = events[:, 1].int()
            t0 = t_norm.int()
            if events.shape[1] == 3:
                value = torch.ones(x0.shape)
            elif events.shape[1] == 4:
                value = 2*events[:, 3]-1
            else:
                raise f"input event columns error! expect 3 or 4, got {events.shape[1]}"

            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim-events[:, 0]).abs()) * (1 - (ylim-events[:, 1]).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask].float(), accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

def show_voxel_grid(voxel_grid, sy_image=260, sx_image=346):
    '''
    voxel_grid: (1, C, H, W)
    '''
    voxel_grid = voxel_grid.cpu()
    images = np.zeros((sy_image, sx_image, 3))
    image = voxel_grid[0][1]
    label_p = (image > 0)
    label_n = (image < 0)
    images[label_p, 0] = image[label_p] * 255
    images[label_n, 2] = abs(image[label_n] * 255)
    images[(~label_p) & (~label_n)] = 255
    return images


def show_voxel_grids(voxel_grid, sy_image=260, sx_image=346):
    '''
    voxel_grid: (1, C, H, W)
    '''
    voxel_grid = voxel_grid.cpu()
    images_list = []
    image = voxel_grid[0]
    for i in range(image.shape[0]):
        images = np.zeros((sy_image, sx_image, 3))
        image_i = image[i]
        label_p = (image_i > 0)
        label_n = (image_i < 0)
        images[label_p, 2] = image_i[label_p] * 255
        images[label_n, 0] = abs(image_i[label_n] * 255)
        images[(~label_p) & (~label_n)] = 255
        voxel = cv2.copyMakeBorder(images, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
        images_list.append(voxel)
    return images_list


if __name__ == '__main__':
    event = torch.tensor([[2,5,0.1,1],[6, 7,0.2,0]], dtype = torch.float)
    voxel = VoxelGrid((4, 10, 10), normalize=True)
    image = voxel.convert(event)
    print(image)