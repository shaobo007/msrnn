import abc
import argparse
import logging
import os
import pytorch_lightning as pl
import torch
import torch.utils.data
import einops

from typing import Callable, List, Optional, Tuple

from .event_ds import EventDataset

class EventDataModule(pl.LightningDataModule):

    def __init__(self, img_shape: Tuple[int, int], batch_size: int, shuffle: bool, num_workers: int,
                 pin_memory: bool, resize: bool, frame_deck: bool, n_frame: int):
        super(EventDataModule, self).__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.resize = resize
        self.frame_deck = frame_deck
        self.n_frame = n_frame

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        logging.info("Preparing datasets for loading")
        self._prepare_dataset("training")
        self._prepare_dataset("validation")
        #self._prepare_dataset("test")

    def setup(self, stage: Optional[str] = None):
        logging.debug("Load and set up datasets")
        self.train_dataset = self._load_dataset("training")
        self.val_dataset = self._load_dataset("validation")
        #self.test_dataset = self._load_dataset("test")
        if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
            print(self.root)
            raise UserWarning("No data found, check AEGNN_DATA_DIR environment variable!")

    #########################################################################################################
    # Data Loaders ##########################################################################################
    #########################################################################################################
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, 
                                           collate_fn=self.collate_fn if self.frame_deck is True else None,
                                           shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=True)

    def val_dataloader(self, batch_size = 1, num_workers: int = 2) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers,
                                           collate_fn=self.collate_fn if self.frame_deck is True else None,
                                           shuffle=False, drop_last=True)

    def test_dataloader(self, num_workers: int = 2) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_dataset, self.batch_size, num_workers=num_workers,
                                           collate_fn=self.collate_fn, shuffle=False)
    #########################################################################################################
    # Processing ############################################################################################
    #########################################################################################################
    @abc.abstractmethod
    def _prepare_dataset(self, mode: str):
        raise NotImplementedError

    @abc.abstractmethod
    def raw_files(self, mode: str) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def processed_files(self, mode: str) -> List[str]:
        raise NotImplementedError

    #########################################################################################################
    # Data Loading ##########################################################################################
    #########################################################################################################
    def _load_dataset(self, mode: str):
        processed_files = self.processed_files(mode)
        logging.debug(f"Loaded dataset with {len(processed_files)} processed files")
        return EventDataset(processed_files, load_func=self.load_processed_file)

    def load_processed_file(self, f_path: str):
        data = self._load_processed_file(f_path)
        voxel_image = data['voxel_image'][:,2:258, 5:341]
        image = data['acc_image'][2:258, 5:341]
        mask = data['mask'][2:258, 5:341]
        depth = data['depth']
        gray = data['gray'][2:258, 5:341]
        class_id = data['class']
        sample = {}
        sample['image'] = voxel_image
        sample['target_size'] = voxel_image.shape
        sample['labels'] = mask.to(dtype=torch.long)
        sample['class'] = class_id
        sample['gray'] = gray
        #return voxel_image, mask, depth
        return sample

    @staticmethod
    def collate_fn(batch):
        images = [data[0] for data in batch]
        images = torch.cat(images,0)
        masks = [data[1] for data in batch]
        masks = torch.cat(masks,0)
        depths = [data[1] for data in batch]
        depths = torch.cat(depths,0)
        return [images, masks, depths]

    @abc.abstractmethod
    def _load_processed_file(self, f_path: str):
        """Load pre-processed file to Data object.

        The pre-processed file is loaded into a torch-geometric Data object. With N the number of events,
        L the number of annotations (e.g., bounding boxes in the sample) and P the number of edges, the
        output object should minimally be as shown below.

        :param f_path: input (absolute) file path of preprocessed file.
        :returns Data(x=[N] (torch.float()), pos=[N, 2] (torch.float()), bbox=[L, 5] (torch.long()), file_id,
                      y=[L] (torch.long()), label=[L] (list), edge_index=[2, P] (torch.long())
        """
        raise NotImplementedError

    #########################################################################################################
    # Dataset Properties ####################################################################################
    #########################################################################################################
    @classmethod
    def add_argparse_args(cls, parent_parser: argparse.ArgumentParser, **kwargs) -> argparse.ArgumentParser:
        parent_parser.add_argument("--dataset", action="store", type=str, required=True)

        group = parent_parser.add_argument_group("Data")
        group.add_argument("--batch-size", action="store", default=8, type=int)
        group.add_argument("--num-workers", action="store", default=8, type=int)
        group.add_argument("--n-frame", action="store", default=2, type=int)
        group.add_argument("--pin-memory", action="store_true")
        group.add_argument("--resize", action="store_true")
        group.add_argument("--frame-deck", action="store_true")
        return parent_parser

    @property
    def root(self) -> str:
        return os.path.join(os.environ["AEGNN_DATA_DIR"], self.__class__.__name__.lower())

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def classes(self) -> List[str]:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __repr__(self):
        train_desc = self.train_dataset.__repr__()
        val_desc = self.val_dataset.__repr__()
        return f"{self.__class__.__name__}[Train: {train_desc}\nValidation: {val_desc}]"
