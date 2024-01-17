"""
Preprocessing Script for DTU Trail dataset
Parsing normal vectors has a large consumption of memory. Please reduce max_workers if memory is limited.
"""

import os
import argparse
import glob
import torch
import numpy as np
import multiprocessing as mp

try:
    import open3d
except ImportError:
    import warnings
    warnings.warn(
        'Please install open3d for parsing normal')

try:
    import trimesh
except ImportError:
    import warnings
    warnings.warn(
        'Please install trimesh for parsing normal')

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

area_mesh_dict = {}


def parse_scean(scean, dataset_root, output_root):
    print("Parsing: {}".format(scean))
    classes = ["unlabeled", "trail", "pavement", "motorway"]
    source_path = os.path.join(dataset_root, scean)
    save_path = os.path.join(output_root, scean).split('.')[0] + ".pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scean_coords = []
    scean_colors = []
    scean_semantic_gt = []
    scean_instance_gt = []

    obj = np.load(source_path)
    coords = obj[:, :3]
    colors = obj[:, 3:6]
    semantic_gt = obj[:, 6]
    semantic_gt = semantic_gt.reshape([-1, 1])
    instance_gt = np.repeat(-1, coords.shape[0])
    instance_gt = instance_gt.reshape([-1, 1])

    scean_coords.append(coords)
    scean_colors.append(colors)
    scean_semantic_gt.append(semantic_gt)
    scean_instance_gt.append(instance_gt)

    scean_coords = np.ascontiguousarray(np.vstack(scean_coords))
    scean_colors = np.ascontiguousarray(np.vstack(scean_colors))
    scean_semantic_gt = np.ascontiguousarray(np.vstack(scean_semantic_gt))
    scean_instance_gt = np.ascontiguousarray(np.vstack(scean_instance_gt))
    save_dict = dict(coord=scean_coords, color=scean_colors, semantic_gt=scean_semantic_gt, instance_gt=scean_instance_gt)
    torch.save(save_dict, save_path)


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='', required=False, help='Path to DTU Trail dataset')
    parser.add_argument('--output_root', default='', required=False, help='Output path where area folders will be located')
    args = parser.parse_args()
    
    args.dataset_root = '/data/dtu_trail/raw'
    args.output_root = '/data/dtu_trail/processed'

    scean_list = []

    # Load scean information
    print("Loading scean information ...")
    data_list = glob.glob(os.path.join(args.dataset_root, "*.npy"))
    for d in data_list:
        scean_list += [d.split('/')[-1]]

    # Preprocess data.
    print('Processing scenes...')
    # pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    pool = ProcessPoolExecutor(max_workers=1)  # peak 110G memory when parsing normal.
    _ = list(pool.map(
        parse_scean, scean_list,
        repeat(args.dataset_root), repeat(args.output_root)
    ))


if __name__ == '__main__':
    main_process()
