import argparse

import torch
import pytorch_lightning as pl

from vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule, IMAGENET_MEAN_STD
from models.backbones.dinov3 import DINOV3_MEAN_STD
from autoconfig import load_config


torch.set_float32_matmul_precision('high')


def parse_config():
    parser = argparse.ArgumentParser(description="Load NN training configuration from a YAML file.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    return config


if __name__ == '__main__':
    config = parse_config()
    print("Config:")
    print(config)

    img_size = config['input_config']['img_size']

    mean_std = DINOV3_MEAN_STD if config['input_config']['mean_std'] == 'DINOV3_MEAN_STD' else IMAGENET_MEAN_STD

    backbone_arch = config['backbone_arch']
    backbone_config = config['backbone_config']
    max_epochs = config['max_epochs']

    datamodule = GSVCitiesDataModule(
        batch_size=60,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=img_size,
        mean_std=mean_std,
        num_workers=10,
        show_data_stats=True,
        val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'],
    )

    model = VPRModel(
        #---- Encoder
        backbone_arch=backbone_arch,
        backbone_config=backbone_config,
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr = 6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='pitts30k_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=5,
        save_last=True,
        mode='max'
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=max_epochs,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
