import json
import os
import random
from copy import deepcopy
from collections import defaultdict
from .utils.lmdb_utils import ImageLmdbReader
from PIL import Image, ImageOps

import torch

class DreamFuseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root,
            meta_root,
            size_list=None,
            name="PatchDataset",
            rank=0,
            worldsize=1,
            lmdb_data=False,
            allow_multi_buckets=False,
            item_num_scale_factor=1.0,
            **kw_args,
    ):
        self.data_root = data_root
        self.size_list = size_list
        self.name = name
        self.rank = rank
        self.worldsize = worldsize
        self.item_num_scale_factor = item_num_scale_factor
        self.allow_multi_buckets = allow_multi_buckets
        self.lmdb_data = lmdb_data
        if self.lmdb_data:
            self.image_lmdb_reader = ImageLmdbReader(self.data_root)

        if os.path.isfile(meta_root) and meta_root.endswith(".json"):
            with open(meta_root) as fr:
                metas = json.load(fr)
            if isinstance(metas, dict):
                metas = list(metas.values())
        else:
            metas = list()
            for _name in sorted([x for x in os.listdir(meta_root) if x.endswith('.json')]):
                with open(os.path.join(meta_root, _name)) as fr:
                    metas += json.load(fr)
        
        self.metas = metas

        # put items in different size buckets
        self.size_to_meta_list = defaultdict(list)
        if size_list is not None:
            target_aspect_ratios = torch.tensor(
                [float(_[0]) / _[1] for _ in size_list]
            )
            for mid, item_meta in enumerate(metas):
                _height, _width = item_meta["height"], item_meta["width"]
                _aspect_ratio = float(_height) / _width
                aspect_ratio_diff = (target_aspect_ratios - _aspect_ratio).abs()
                if aspect_ratio_diff.min() < 0.3:
                    _same_aspect_ratio = torch.where(aspect_ratio_diff <= 1e-2)[
                        0
                    ].tolist()
                    if len(_same_aspect_ratio) > 0:
                        candidate_bucket_indices = _same_aspect_ratio
                    else:
                        candidate_bucket_indices = [aspect_ratio_diff.argmin().item()]

                    if self.allow_multi_buckets:
                        for selected_bucket_index in candidate_bucket_indices:
                            target_height, target_width = size_list[selected_bucket_index]
                            self.size_to_meta_list[(target_height, target_width)].append(mid)
                    else:
                        selected_bucket_index = candidate_bucket_indices[0]
                        min_height_diff = abs(
                            size_list[selected_bucket_index][0] - _height
                        )
                        for _idx in candidate_bucket_indices:
                            height_diff = abs(size_list[_idx][0] - _height)
                            if height_diff < min_height_diff:
                                height_diff = min_height_diff
                                selected_bucket_index = _idx
                        target_height, target_width = size_list[selected_bucket_index]
                        self.size_to_meta_list[(target_height, target_width)].append(mid)

        else:
            self.size_to_meta_list[None] = metas

        self.reset_local_meta_list_()

    def __len__(self):
        return int(self.total_len * self.item_num_scale_factor)

    def reset_local_meta_list_(self):
        self.total_len = 0
        self.local_size_to_meta_list = dict()
        for key in self.size_to_meta_list:
            meta_list = self.size_to_meta_list[key]
            if len(meta_list) < self.worldsize:
                print(
                    f"Ignore {len(meta_list)} items with target size {key} in {self.name}\n"
                )
            else:
                meta_list = deepcopy(meta_list)
                random.shuffle(meta_list)
                part_size = len(meta_list) // self.worldsize
                self.local_size_to_meta_list[key] = meta_list[
                                                    self.rank * part_size: (self.rank + 1) * part_size
                                                    ]
                self.total_len += len(self.local_size_to_meta_list[key])

    def get_item_num_by_size(self, size):
        if size not in self.local_size_to_meta_list:
            return 0
        else:
            return int(
                len(self.local_size_to_meta_list[size]) * self.item_num_scale_factor
            )

    def get_item_meta_by_size(self, size):
        meta_list = self.local_size_to_meta_list[size]
        return random.choice(meta_list)

    def get_image(self, data_key):
        if self.lmdb_data:
            image = self.image_lmdb_reader(data_key)
        else:
            image_path = os.path.join(self.data_root, data_key)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
            image = Image.open(image_path)
        return image

    def __getitem__(self, target_size):
        item_meta = self.metas[self.get_item_meta_by_size(target_size)]
        keys = item_meta["sorted_keys"]
        prompts = [item_meta['caption_info'][key] for key in keys]
        images = [self.get_image(item_meta['img_info'][key]) for key in keys]
        mask_images = [
            self.get_image(item_meta['img_mask_info']['000_mask_scale']).convert('L'),
            self.get_image(item_meta['img_mask_info']['000']).convert('L'),
            self.get_image(item_meta['img_mask_info']['000_paste']),
        ]

        # make mask white
        images[0].paste((255, 255, 255), mask=ImageOps.invert(mask_images[1]))

        return dict(
            prompts=prompts,
            images=images,
            mask_images=mask_images,
            # pasted_dpo="paste" in item_meta["img_mask_info"]["000_paste"]
        )