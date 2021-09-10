# Python std
import argparse
import os
from timeit import default_timer as timer

# 3rd party
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# project files
from dataset import TempFramePairs, TempFramePairsFixed
from helpers import collat_fn, collate_feats_with_none_dict, weights_init_uniform
from conf import add_arguments
from models.network import AE_AtlasNet, AE_AtlasNet5D, AE_AtlasNetInPlace
from models.affine_network import AffineNet, AffineMLPNet, RotMLPNet, IdentityNet
from models.fix_mapping_network import FixMapAffineMLPNet





def cli_main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_arguments(parser)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    # make dataloader
    train_dataset = TempFramePairs(data_sample=args.data_sample, sample_size=args.sample_size, data_len=args.data_len)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, drop_last=True, shuffle=False) #, collate_fn=collate_feats_with_none_dict)
    val_loader = DataLoader(train_dataset)

    logger = TensorBoardLogger("tb_logs", name=args.output_dir.split('/')[-1])


    # use different networks
    if args.affine_net:
        network = AffineNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim)
        network = network.cuda(device=args.gpu) 

    elif args.affinemlp_net:
        network = AffineMLPNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim)
        network = network.cuda(device=args.gpu) 

    elif args.rotmlp_net:
        network = RotMLPNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim)
        network = network.cuda(device=args.gpu) 
    
    elif args.identity_net:
        network = IdentityNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim)
        network = network.cuda(device=args.gpu) 

    elif args.inplace_net:
        network = AE_AtlasNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim)
        network = network.cuda(device=args.gpu) 
    
    elif args.fixmap_net:
        train_dataset2 = TempFramePairsFixed(data_sample=args.data_sample, sample_size=args.sample_size, data_len=args.data_len)
        train_loader2 = DataLoader(train_dataset2, batch_size = args.batch_size, drop_last=True, shuffle=False) #, collate_fn=collate_feats_with_none_dict)
        val_loader2 = DataLoader(train_dataset2)
        points = next(iter(train_loader2))['pts1'][0].cuda()
        network = FixMapAffineMLPNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim, points=points)
        network = network.cuda(device=args.gpu) 
        trainer = pl.Trainer.from_argparse_args(args, gpus=[args.gpu], progress_bar_refresh_rate=5, max_epochs=args.epochs, logger=logger)
        trainer.fit(network, train_loader2, val_loader2)
    
    elif args.atlas5d_net:
        network = AE_AtlasNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim)
        network = network.cuda(device=args.gpu) 

    else:
        network = AE_AtlasNet(args, learning_rate=args.lr, batch_size=args.batch_size, num_points = args.num_points, nb_primitives = args.nb_primitives, output_dim=args.output_dim)
        network = network.cuda(device=args.gpu) 

    trainer = pl.Trainer.from_argparse_args(args, gpus=[args.gpu], progress_bar_refresh_rate=5, max_epochs=args.epochs, logger=logger)
    trainer.fit(network, train_loader, val_loader)


if __name__ == '__main__':
    cli_main()