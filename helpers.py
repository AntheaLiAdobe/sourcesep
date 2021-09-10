# Python std
import os
import math
import shutil

# 3rd party
import yaml
import numpy as np
import torch
import torch.nn as nn

# Project files.
import helpers as helpers
    

def load_conf(path):
    """ Returns the loaded .cfg config file.

    Args:
        path (str): Aboslute path to .cfg file.

    Returns:
    dict: Loaded config file.
    """

    with open(path, 'r') as f:
        conf = yaml.full_load(f)
    return conf


def save_conf(conf, fname, key_path_trrun='path_train_run', force_base_dir_perm=False):
    """ Creates the output path and saves the config.
    """
    # Get train run path.
    trrun_subdir = create_trrun_name(conf)
    out_path = helpers.jn(conf[key_path_trrun], trrun_subdir)
    base_dir_exists = os.path.exists(conf[key_path_trrun])

    # Create train run dir.
    if os.path.exists(out_path):
        out_path_old = out_path
        out_path = helpers.unique_dir_name(out_path)
        print('WARNING: The output path {} already exists, creating new dir {}'.format(out_path_old, out_path))
    helpers.make_dir(out_path)
    if force_base_dir_perm:
        os.chmod(out_path, 0o0777)
        if not base_dir_exists:
            os.chmod(conf[key_path_trrun], 0o0777)

    # Save config.
    with open(helpers.jn(out_path, fname), 'w') as f:
        yaml.dump(conf, f)

    return out_path


def collate_feats_with_none_list(b):
    b = filter (lambda x:x is not None, b)
    return list(zip(*b))


def collate_feats_with_none_dict(b):
    b = filter (lambda x:x is not None, b)

    print(dict(zip(*b)))
    return dict(zip(*b))

def collat_fn(batch):
    bc = {k: [] for k in batch[0].keys()}
    for it in batch:
        for k, v in it.items():
            bc[k].append(v)
    return {k: torch.cat(v, dim=0) for k, v in bc.items()}


def weights_init_uniform(m):
    for module in m.parameters():
    #   if isinstance(m, nn.Conv1d):
    #       nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
    #       if m.bias is not None:
    #           nn.init.constant_(m.bias.data, 0)
    #   elif isinstance(m, nn.BatchNorm2d):
    #       nn.init.constant_(m.weight.data, 1)
    #       nn.init.constant_(m.bias.data, 0)
        print(module)
        if isinstance(module, nn.Linear):
            print('here')
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1.0)
            #   nn.init.kaiming_uniform_(m.weight.data)
            #   nn.init.constant_(m.bias.data, 0)



def create_trrun_save_conf(path_conf, key_path_trrun='path_train_run',
                           force_base_dir_perm=False, ds=None,
                           ds_specific_path=False):
    """ Loads the configuration file (.yaml), creates a new training run output
    dir, saves the config. file into the output dir, returns the config and the
    out path.

    Args:
        path_conf (str): Absolute path to the configuration file.
        key_run (str): Dict key to value storing path to the dir holding
            trianing data.

    Returns:
        conf (dict): Loaded config file.
        out_path (str): Path to new output dir.
    """

    # Load conf.
    conf = load_conf(path_conf)
    # out_path = save_conf(
    #     conf, os.path.basename(path_conf), key_path_trrun=key_path_trrun,
    #     force_base_dir_perm=force_base_dir_perm)

    # Get train run path.
    trrun_subdir = create_trrun_name(conf, ds=ds)
    out_path = helpers.jn(conf[key_path_trrun], trrun_subdir)
    if ds_specific_path:
        assert ds is not None
        if ds in ('dfaust', 'ama', 'animals', 'cape', 'inria', 'cmu'):
            seq = conf['sequences']
            assert isinstance(seq, (str, list, tuple))
            if isinstance(seq, (list, tuple)):
                assert len(seq) == 1
                seq_str = seq[0]
            elif isinstance(seq, str):
                assert seq == 'all'
                seq_str = 'all'
            out_path = helpers.jn(conf[key_path_trrun], seq_str, trrun_subdir)
        else:
            raise Exception(f"Unknown ds {ds}.")
    base_dir_exists = os.path.exists(conf[key_path_trrun])

    # Create train run dir.
    if os.path.exists(out_path):
        out_path_old = out_path
        out_path = helpers.unique_dir_name(out_path)
        print('WARNING: The output path {} already exists, creating new dir {}'.
              format(out_path_old, out_path))
    helpers.make_dir(out_path)
    if force_base_dir_perm:
        os.chmod(out_path, 0o0777)
        if not base_dir_exists:
            os.chmod(conf[key_path_trrun], 0o0777)

    # Save config.
    shutil.copy(path_conf, helpers.jn(out_path, os.path.basename(path_conf)))

    return conf, out_path


def prepare_uv(num_pts, num_patches):
    """ Generates points spaced in a regular grid in 2D space. If
    `num_pts` cannot be divided into `num_patches` P so that each patch
    would have E x E pts, `num_pts` is adjusted to the closest number
    E ** 2 * num_patches. Every patch thus gets exactly the same set
    of 2D point coordinates.

    Args:
        num_pts (int): # points to generate.
        num_patches (int): # patches the model uses.

    Returns:
        np.array[float32]: Points, (N, 2), N = E ** 2 * P.
        int: Adjusted # sampled points.
    """
    ppp = num_pts / num_patches
    ev = int(round(math.sqrt(ppp)))
    M = int(ev ** 2 * num_patches)
    if M != num_pts:
        print(f"[WARNING]: Cannot split {num_pts} among {num_patches} patches "
              f"regularly, using {M} instead ({ev ** 2} = {ev} * {ev} "
              f"pts per patch).")
    return np.tile(helpers.grid_verts_2d(ev, ev, 1., 1.), (num_patches, 1)), M


class LRSchedulerFixed:
    def __init__(self, opt, iters, lrfr, verbose=True):
        assert isinstance(iters, (int, list, tuple))
        assert isinstance(lrfr, (float, list, tuple))

        if isinstance(iters, int):
            iters = [iters]
        if isinstance(lrfr, float):
            lrfr = [lrfr] * len(iters)
        assert len(iters) == len(lrfr)

        self._step = 0
        self._opt = opt
        self._iters = np.array(iters)
        self._lrfr = lrfr
        self._verbose = verbose

    def step(self):
        self._step += 1
        it = np.where(self._step == self._iters)[0]
        assert len(it) <= 1
        if len(it) == 1:
            lf = self._lrfr[it[0]]
            lrold = self._opt.param_groups[0]['lr']
            lrnew = lrold * lf
            if self._verbose:
                print(f"[INFO] Reached iter {self._step} and "
                      f"changing lr from {lrold} to {lrnew}.")
            self._opt.param_groups[0]['lr'] = lrnew

    def state_dict(self):
        return {'step': self._step}

    def load_state_dict(self, d):
        self._step = d['step']
