import argparse

import torch
import pytorch_lightning as pl

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from utils.yaml_config import load_config
from vpr.models import get_transforms, VPRModel


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

    backbone_arch = config['backbone_arch']
    input_config = config['input_config']
    backbone_config = config['backbone_config']
    agg_config = config['agg_config']
    max_epochs = config['max_epochs']

    train_transform, valid_transform = get_transforms(backbone_arch, input_config)

    datamodule = GSVCitiesDataModule(
        train_transform,
        valid_transform,
        batch_size=60,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        num_workers=10,
        show_data_stats=True,
        #Bug: https://github.com/L4rralde/Visual_Place_Recognition/issues/1?reload=1
        #val_set_names=['pitts30k_val', 'pitts30k_test', 'msls_val'], 
        val_set_names=['pitts30k_val', 'pitts30k_test'], #FIXME. By the moment, the transformation used for da3dino does not work with msls_val because there are images with different aspect ratios
    )

    model = VPRModel(
        #---- Encoder
        backbone_arch=backbone_arch,
        backbone_config=backbone_config,
        agg_arch='SALAD',
        agg_config=agg_config,
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
