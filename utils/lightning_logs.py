import os
from typing import Dict
import re
import yaml

import pandas as pd
import torch


class LightningLog:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conf = LightningLog.load_config(os.path.join(path, 'hparams.yaml'))

    @staticmethod
    def load_config(path: str) -> None:
        with open(path, 'r') as f:
            content = f.read()
        
        clean_content = re.sub(
            r'!!python/(object|apply|set)[:/][^\s]+',
            '',
            content
        )
        config = yaml.safe_load(clean_content)

        required_fields = ['backbone_arch',  'backbone_config', 'agg_config']
        for field in required_fields:
            if not field in config:
                raise RuntimeError(f"Expected: {field} in yaml file: {path}")
            
        return config

    @property
    def backbone_arch(self) -> str:
        return self.conf['backbone_arch']

    @property
    def agg_arch(self) -> str:
        return self.conf['agg_arch']

    @property
    def backbone_config(self) -> dict:
        return self.conf['backbone_config']

    @property
    def agg_config(self) -> dict:
        return self.conf['agg_config']

    @staticmethod
    def load_metrics(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @property
    def metrics(self) -> pd.DataFrame:
        return LightningLog.load_metrics(
            os.path.join(self.path, 'metrics.csv')
        )

    @property
    def validation_results(self) -> pd.DataFrame:
        df = self.metrics

        r_columns = [
            col for col in df.columns
            if re.search(r'/R\d+$', col)
        ]

        if not r_columns:
            return df.iloc[0:0]  # empty DataFrame with same columns

        mask = df[r_columns].notna().any(axis=1)

        return df[mask].dropna(axis=1, how='all').reset_index(drop=True)


    @staticmethod
    def load_ckpt(path: str) -> Dict[str, torch.Tensor]:
        return torch.load(
            path,
            weights_only=False,
            map_location=torch.device('cpu')
        )['state_dict']

    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return LightningLog.load_ckpt(
            os.path.join(self.path, 'checkpoints', 'last.ckpt')
        )
