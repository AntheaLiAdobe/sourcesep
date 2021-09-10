import os 
import numpy as np
from argparse import ArgumentParser




def add_arguments(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # general setup 
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--weight_path', type=str, default='',  help='optional reload model path')
    parser.add_argument('--env', type=str, default="AE_AtlasNet"   ,  help='visdom environment')
    parser.add_argument('--output_dir', type=str, default="./output/output1/"   ,  help='save_directory')
    parser.add_argument('--vis_freq', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--save_freq', type=int, default=3000, help='number of epochs to train for')


    # network spec and loss parameters
    parser.add_argument('--decoder_activation', type=str, default='sigmoid',  help='use sigmoid because it is now normalized to 0 ~ 1')
    parser.add_argument('--shape_activation', type=str, default='tanh',  help='use sigmoid because it is now normalized to 0 ~ 1')
    parser.add_argument('--color_activation', type=str, default='sigmoid',  help='use sigmoid because it is now normalized to 0 ~ 1')
    parser.add_argument('--sample_size', type=int, default=1500,  help='number of points')
    parser.add_argument('--num_points', type=int, default=2500,  help='number of points')
    parser.add_argument('--output_dim', type=int, default=5,  help='number of output dimension')
    parser.add_argument('--nb_primitives', type=int, default=1,  help='number of primitives in the atlas')
    parser.add_argument('--super_points', type=int, default=2500,  help='number of input points to pointNet, not used by default')
    parser.add_argument('--no_color_loss',  default=False,  action='store_true', help='whether to use color loss')
    parser.add_argument('--no_temporal_loss', default=False, action='store_true',  help='whether to use temporal loss')
    parser.add_argument('--chamfer5d_loss',  default=False,  action='store_true', help='whether to use chamfer5d for temporal loss')
    parser.add_argument('--jacobian',  default=False,  action='store_true', help='whether to use chamfer5d for temporal loss')
    parser.add_argument('--shape_loss_weight', default=1.0, type=float)
    parser.add_argument('--color_loss_weight', default=1.0, type=float)
    parser.add_argument('--shape_temporal_loss_weight', default=1.0, type=float)
    parser.add_argument('--color_temporal_loss_weight', default=1.0, type=float)
    parser.add_argument('--temporal_loss_weight', default=1.0, type=float)
    parser.add_argument('--shape_color_scale', default=1.0, type=float)

    

    # temporal experiment specifications
    parser.add_argument('--data_sample', type=int, default=1,  help='data_sample')
    parser.add_argument('--data_len', type=int, default=1,  help='data_sample')
    parser.add_argument('--affine_net',  default=False,  action='store_true', help='affine trans')
    parser.add_argument('--affinemlp_net',  default=False,  action='store_true', help='affine trans')
    parser.add_argument('--rotmlp_net',  default=False,  action='store_true', help='affine trans')
    parser.add_argument('--fixmap_net',  default=False,  action='store_true', help='affine trans')
    parser.add_argument('--identity_net',  default=False,  action='store_true', help='affine trans')
    parser.add_argument('--inplace_net',  default=False,  action='store_true', help='affine trans')
    parser.add_argument('--atlas5d_net',  default=False,  action='store_true', help='affine trans')



    return parser
