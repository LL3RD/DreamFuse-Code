import random
import torch.utils
import torch.utils.data
import yaml
from PIL import Image
from collections import defaultdict

import torch
import numpy as np

from .stubs.dreamfuse_data import DreamFuseDataset


class MergedEditDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets, size_list, batchsize, rank, worldsize, use_ratio):
        super().__init__()

        self.batchsize = batchsize
        self.rank = rank
        self.worldsize = worldsize
        self.use_ratio = use_ratio

        self.datasets = datasets
        self.total_len = sum([len(_["db"]) for _ in datasets])
        self.size_to_dbs, size_probs, valid_sizes = dict(), list(), list()
        for size in size_list:
            valid_dbs = [_ for _ in datasets if _["db"].get_item_num_by_size(size) > 0]
            if len(valid_dbs) > 0:
                if use_ratio:
                    db_probs = np.array([_["ratio"] for _ in valid_dbs])
                else:
                    db_probs = np.array(
                        [_["db"].get_item_num_by_size(size) for _ in valid_dbs]
                    )
                db_probs = np.cumsum(db_probs / db_probs.sum())
                self.size_to_dbs[size] = dict(
                    dbs=[_["db"] for _ in valid_dbs], probs=db_probs
                )
                valid_sizes.append(size)
                size_probs.append(
                    sum([_["db"].get_item_num_by_size(size) for _ in datasets])
                )
        size_probs = np.array(size_probs)
        self.size_probs = np.cumsum(size_probs / size_probs.sum())
        self.size_list = valid_sizes

    def __repr__(self):
        info_str = (
            f"MergedEditDataset: rank {self.rank} of worldsize {self.worldsize}\n"
        )
        for size in self.size_to_dbs:
            dbs, db_probs = (
                self.size_to_dbs[size]["dbs"],
                self.size_to_dbs[size]["probs"],
            )
            info_str += f"  {size}:\n"
            for _db, _prob in zip(dbs, db_probs):
                info_str += (
                    f"    {_db.name}, {_db.get_item_num_by_size(size)}, {_prob:.3f}\n"
                )

        return info_str

    def reset_local_meta_list_(self):
        for db in self.datasets:
            db["db"].reset_local_meta_list_()

    def _resize_image_to_target_size(self, item, target_size, keep_aspect_ratio=False):
        target_height, target_width = target_size
        target_aspect_ratio = float(target_height) / target_width

        image = item["images"][0]
        image_height, image_width = image.height, image.width
        image_aspect_ratio = float(image_height) / image_width
        if keep_aspect_ratio:
            if image_aspect_ratio >= target_aspect_ratio:
                resized_image_width = target_width
                resized_image_height = int(
                    round(image_height * (target_width / image_width))
                )
                crop_top = random.randint(0, resized_image_height - target_height)
                crop_left = 0
            else:
                resized_image_height = target_height
                resized_image_width = int(
                    round(image_width * (target_height / image_height))
                )
                crop_top = 0
                crop_left = random.randint(0, resized_image_width - target_width)
        else:
            # Avoid clipping of the editing area
            resized_image_width, resized_image_height = target_width, target_height
            crop_top, crop_left = 0, 0

        for k in range(len(item['images'])):
            item['images'][k] = (
                item['images'][k]
                .resize(
                    (resized_image_width, resized_image_height), resample=Image.BILINEAR
                )
                .crop(
                    (
                        crop_left,
                        crop_top,
                        crop_left + target_width,
                        crop_top + target_height,
                    )
                )
            )
            
        if "mask_images" in item:
            for k in range(len(item['mask_images'])):
                item['mask_images'][k] = (
                item['mask_images'][k]
                .resize(
                    (resized_image_width, resized_image_height), resample=Image.BILINEAR
                )
                .crop(
                    (
                        crop_left,
                        crop_top,
                        crop_left + target_width,
                        crop_top + target_height,
                    )
                )
            )

    def __iter__(self):
        while True:
            # sample size
            size_index = np.searchsorted(self.size_probs, random.random())
            size_index = min(size_index, self.size_probs.size - 1)
            target_size = self.size_list[size_index]

            batch = defaultdict(list)
            for _ in range(self.batchsize):
                # sample dataset
                dbs, db_probs = (
                    self.size_to_dbs[target_size]["dbs"],
                    self.size_to_dbs[target_size]["probs"],
                )
                db_index = np.searchsorted(db_probs, random.random())
                db_index = min(db_index, db_probs.size - 1)
                item = dbs[db_index][target_size]
                self._resize_image_to_target_size(item, target_size)
                for k in item:
                    batch[k].append(item[k])
            for k in batch:
                assert (
                    len(batch[k]) == self.batchsize
                ), f"{k}: has {len(batch[k])} elements, but batchsize is {self.batchsize}"
            yield batch

    def __len__(self):
        return self.total_len


def create_merged_dataset(path, batchsize, rank=0, worldsize=1, use_ratio=False):
    DB_TYPE_MAP = dict(
        DreamFuseDataset=DreamFuseDataset,
    )
    data_configs = read_yaml(path)
    datasets = list()
    size_list = [tuple(_) for _ in data_configs["size_list"]]
    for db_config in data_configs["datasets"]:
        db_type = db_config.pop("dataset_type")
        if db_type in DB_TYPE_MAP:
            ratio = db_config.pop("ratio", None)
            db_config.update(
                dict(
                    rank=rank,
                    worldsize=worldsize,
                    name=db_config.get("name", None) or db_type,
                    size_list=size_list,
                )
            )
            db = DB_TYPE_MAP[db_type](**db_config)
            datasets.append(dict(ratio=ratio, db=db))
        else:
            print("Ignore unknow dataset type: {}".format(db_type))
    return MergedEditDataset(datasets, size_list, batchsize, rank, worldsize, use_ratio)

def read_yaml(path):
    with open(path) as fr:
        return yaml.safe_load(fr)


if __name__ == "__main__":
    import torch

    path = "../../../data/ic_data.yaml"
    mdb = create_merged_ic_dataset(
        path=path, batchsize=8, rank=0, worldsize=1, use_ratio=False
    )
    print(mdb)
    print(len(mdb))

    dataloader = torch.utils.data.DataLoader(
        dataset=mdb,
        batch_size=None,
        num_workers=4,
        prefetch_factor=16,
        pin_memory=False,
        collate_fn=None,
    )

    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    print(batch)
