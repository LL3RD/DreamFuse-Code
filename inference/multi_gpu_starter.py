import os
import torch
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from dataclasses import dataclass, field
import transformers

@dataclass
class TrainConfig:
    gpu_num: int = 8
    total_num: int = 8
    start_cnt: int = 0

def main():
    parser = transformers.HfArgumentParser(TrainConfig)
    config: TrainConfig = parser.parse_args_into_dataclasses()[0]
    with tqdm(range(config.gpu_num)) as pbar:
        def map_func(gpu_id):
            sub_idx = gpu_id + config.start_cnt
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id % config.gpu_num} python3 inference/dreamfuse_inference.py --sub_idx {sub_idx} --total_num {config.total_num}'
            os.system(command=cmd)
            pbar.update()

        pool = ThreadPool(config.gpu_num)
        pool.map(map_func, range(config.gpu_num))
        pool.close()

if __name__ == '__main__':
    main()