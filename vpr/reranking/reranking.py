import os
from typing import List, Dict
from copy import copy
from tqdm import tqdm
from shutil import rmtree

from torch.utils.data import Dataset
import numpy as np

from .lightglue import LightGlue
from .superpoint import SuperPoint


class Reranker:
    def __init__(
        self, val_dataset: Dataset,
        dump_dir: os.PathLike='./tmp/superpoint/'
    ) -> None:
        self.val_dataset: Dataset = copy(val_dataset)
        self.val_dataset.transform = None
        self.dump_dir: os.PathLike = dump_dir
        rmtree(self.dump_dir)
        os.makedirs(self.dump_dir)
        self.lightglue = LightGlue()

    def extract_local_features(self) -> None:
        feats_dir = os.path.join(self.dump_dir, 'features')
        os.makedirs(feats_dir)
        print("Extracting descriptors")
        sp = SuperPoint()
        for img, idx in tqdm(self.val_dataset):
            feats = sp.run([img])[0]
            path = os.path.join(feats_dir, f'{idx}.npz')
            np.savez(path, **feats)
    
    def get_local_features(self, idx: int) -> Dict:
        feats_dir = os.path.join(self.dump_dir, 'features')
        path = os.path.join(feats_dir, f'{idx}.npz')
        return np.load(path)
    
    def rerank(self, query_idx: int, topk_idcs: List[int]) -> List[int]:
        query_local_feats = self.get_local_features(query_idx)
        inliers_cnt = [
            self.lightglue.get_inliers_count(
                query_local_feats,
                self.get_local_features(ref_idx)
            )
            for ref_idx in topk_idcs
        ]

        ranks = sorted(
            range(len(inliers_cnt)),
            key=lambda i: inliers_cnt[i],
            reverse=True
        )

        return {
            'permutation': ranks,
            'inliers_cnt': inliers_cnt
        }

            