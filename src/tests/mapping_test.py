import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.util.grid import pcd2grid
from src.libs.HybridAStar.hybrid_a_star import hybrid_a_star_planning
from src.libs.HybridAStar.car import rectangle_check, plot_car


def plan_path(nodes, ranges, grid_size):
    ox = np.hstack((
        np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size),
        np.array([ranges['x_min'] - 0.5] * np.arange(-1.5, ranges['y_max'] + 1.5, grid_size).shape[0]),
        np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size),
        np.array([ranges['x_max'] + 0.5] * np.arange(-1.5, ranges['y_max'] + 1.5, grid_size).shape[0])
    ))
    oy = np.hstack((
        np.array([-1.5] * np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size).shape[0]),
        np.arange(-1.5, ranges['y_max'] + 1.5, grid_size),
        np.array([ranges['y_max'] + 1.5] * np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size).shape[0]),
        np.arange(-1.5, ranges['y_max'] + 1.5, grid_size)
    ))
    ox, oy = list(np.around(ox, 2)), list(np.around(oy, 2))
    if len(nodes['obstacles']):
        ox += nodes['obstacles'][0]
        oy += nodes['obstacles'][1]
    if len(nodes['motorway']):
        ox += nodes['motorway'][0]
        oy += nodes['motorway'][1]
    
    x = np.array(nodes['trail'][0])
    y = np.array(nodes['trail'][1])
    tx = np.around(x / grid_size, 0)
    ty = np.around(y / grid_size, 0)
    px = np.around(np.array(nodes['pavement'][0]) / grid_size, 0)
    py = np.around(np.array(nodes['pavement'][1]) / grid_size, 0)
    otx = np.around(np.array(nodes['others'][0]) / grid_size, 0)
    oty = np.around(np.array(nodes['others'][1]) / grid_size, 0)
    can_pass = [tx, ty, px, py, otx, oty]

    # Set Initial parameters
    start = [0.0, -1.2, np.deg2rad(90.0)]
    uni_y = np.unique(y)
    countx = np.array([len(x[y == uy]) for uy in uni_y])
    uni_y = uni_y[countx > 2]
    if not len(uni_y):
        return None
    # uni_y = uni_y[(np.max(uni_y) - uni_y) > 1.0]
    # if not len(uni_y):
    #     return None
    sorty_idx = np.argsort(uni_y)
    i = 1
    while i < sorty_idx.shape[0]:
        # if i > int(sorty_idx.shape[0] / 2) and sorty_idx.shape[0] >= 8:
        #     return None
        idx = sorty_idx[-i]
        gy = uni_y[idx]
        dx = np.abs(x[y == gy] - np.mean(x[y == gy]))
        sortx_idx = np.argsort(dx)
        gx = x[y == gy][sortx_idx[0]]
        if gx != 0:
            theta = np.arctan(gy / gx)
            if theta < 0:
                theta = np.pi + theta
        else:
            theta = np.pi / 2
        if rectangle_check(gx, gy, theta, ox, oy):
            goal = [gx, gy, theta]
            path = hybrid_a_star_planning(can_pass, start, goal, ox, oy, xy_resolution=grid_size, yaw_resolution=np.deg2rad(5.0))
            if path:
                return path
        i += 1
    return None


def plot_map(nodes):
    plt.figure(figsize=(10, 10), layout='constrained')
    if len(nodes['obstacles']):
        plt.scatter(np.array(nodes['obstacles'][0]), np.array(nodes['obstacles'][1]), c='black', s=10)
    if len(nodes['others']):
        plt.scatter(np.array(nodes['others'][0]), np.array(nodes['others'][1]), c='b', s=10)
    if len(nodes['trail']):
        plt.scatter(np.array(nodes['trail'][0]), np.array(nodes['trail'][1]), c='r', s=10)
    if len(nodes['pavement']):
        plt.scatter(np.array(nodes['pavement'][0]), np.array(nodes['pavement'][1]), c='y', s=10)
    if len(nodes['motorway']):
        plt.scatter(np.array(nodes['motorway'][0]), np.array(nodes['motorway'][1]), c='lime', s=10)
    plt.grid(True)
    plt.show()


def plot_path(nodes, path):
    plt.figure(figsize=(10, 10), layout='constrained')
    if len(nodes['obstacles']):
        plt.scatter(np.array(nodes['obstacles'][0]), np.array(nodes['obstacles'][1]), c='black', s=10)
    if len(nodes['others']):
        plt.scatter(np.array(nodes['others'][0]), np.array(nodes['others'][1]), c='b', s=10)
    if len(nodes['trail']):
        plt.scatter(np.array(nodes['trail'][0]), np.array(nodes['trail'][1]), c='r', s=10)
    if len(nodes['pavement']):
        plt.scatter(np.array(nodes['pavement'][0]), np.array(nodes['pavement'][1]), c='y', s=10)
    if len(nodes['motorway']):
        plt.scatter(np.array(nodes['motorway'][0]), np.array(nodes['motorway'][1]), c='lime', s=10)
    if path is not None:
        x = np.array(path.x_list)
        y = np.array(path.y_list)
        plt.plot(x, y, "-k", label="Hybrid A* path")
        plot_car(x[0], y[0], path.yaw_list[0])
    plt.grid(True)
    plt.show()


file_names = ['20230315_145631_5495', '20230315_145631_5385', '20230315_145631_9500', '20230315_145631_8290', '20230315_145631_16670']

for file_name in file_names:
    pcd_path = '/mnt/Documents/DTU/Thesis/report/presented_data/whole_pcd/{}.ply'.format(file_name)
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    grid_size = 0.3

    nodes, ranges = pcd2grid(pcd, grid_size=grid_size)
    path = plan_path(nodes, ranges, grid_size)
    
    plot_map(nodes)
    plot_path(nodes, path)

print('end')
