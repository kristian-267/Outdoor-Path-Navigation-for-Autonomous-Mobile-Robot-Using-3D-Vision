import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import math
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from configs.navigation.navigation import model_cfg, voxelize_cfg, path_cfg, camera_cfg, grid_cfg
from src.utils.voxelize import Voxelize
from src.utils.ransac import find_ground
from src.utils.filters import set_filters, filter_depth
from src.utils.pointclouds import generate_pcd, convert2pcd, prediction2pcd
from src.utils.inference import inference
from src.utils.model import load_model
from src.utils.grid import pcd2grid
from src.utils.image import cat_color_depth
from src.utils.visualization import visualize
from src.libs.HybridAStar.hybrid_a_star import hybrid_a_star_planning
from src.libs.HybridAStar.car import rectangle_check
from src.dataset.data_prepare import center_shift


def plan_path(nodes, ranges, grid_size):
    '''
    config = Config()
    config.robot_type=RobotType.rectangle
    ox = np.hstack((
        np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size),
        np.array([ranges['x_min'] - 0.5] * np.arange(-1.5, ranges['y_max'] + 0.5, grid_size).shape[0]),
        np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size),
        np.array([ranges['x_max'] + 0.5] * np.arange(-1.5, ranges['y_max'] + 0.5, grid_size).shape[0])
    ))
    oy = np.hstack((
        np.array([-1.5] * np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size).shape[0]),
        np.arange(-1.5, ranges['y_max'] + 0.5, grid_size),
        np.array([ranges['y_max'] + 1] * np.arange(ranges['x_min'] - 0.5, ranges['x_max'] + 0.5, grid_size).shape[0]),
        np.arange(-1.5, ranges['y_max'] + 0.5, grid_size)
    ))
    ob = np.column_stack((np.around(ox, 2), np.around(oy, 2)))
    if len(nodes['obstacles']):
        ob = np.vstack((ob, np.column_stack((np.array(nodes['obstacles'][0]), np.array(nodes['obstacles'][1])))))
    if len(nodes['motorway']):
        ob = np.vstack((ob, np.column_stack((np.array(nodes['motorway'][0]), np.array(nodes['motorway'][1])))))
    config.ob = ob

    tx = np.array(nodes['trail'][0])
    ty = np.array(nodes['trail'][1])
    px = np.array(nodes['pavement'][0])
    py = np.array(nodes['pavement'][1])
    otx = np.array(nodes['others'][0])
    oty = np.array(nodes['others'][1])
    can_pass = [tx, ty, px, py, otx, oty]

    # Set Initial parameters
    x = np.array([0.0, -0.6, np.pi / 2, 0.0, 0.0])
    uniqy = np.unique(ty)
    countx = np.array([len(tx[ty == y]) for y in uniqy])
    uniqy = uniqy[countx > 3]
    sorty = np.argsort(uniqy)
    gy = uniqy[sorty[-1]]
    dx = np.abs(tx[ty == gy] - np.mean(tx[ty == gy]))
    gx = tx[ty == gy][np.argsort(dx)[0]]
    if gx != 0:
        theta = np.arctan(gy / gx)
        if theta < 0:
            theta = np.pi + theta
    else:
        theta = np.pi / 2
    goal = np.array([gx, gy, theta, 0.0, 0.0])
    
    trajectory = np.array(x)
    show_animation = True
    i = 0
    while i <= 3:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            return trajectory
        i += 1
    return trajectory
    '''
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

def get_command(path):
    move_x = np.around(np.array(path.x_list[1:]) - np.array(path.x_list[:-1]), 2)
    move_y = np.around(np.array(path.y_list[1:]) - np.array(path.y_list[:-1]), 2)
    distance = np.around(np.sqrt(move_x ** 2 + move_y ** 2), 1)
    angle = np.around(np.array(path.yaw_list) / (2 * np.pi) * 360, 0)
    turn_angle = list()
    move = list()
    flag = 1
    for i in range(1, angle.shape[0]):
        if angle[i] != angle[i - 1]:
            flag = 1
            turn_angle.append(angle[i] - angle[i - 1])
            move.append(distance[i-1])
        else:
            if flag:
                turn_angle.append(angle[i] - angle[i - 1])
                move.append(distance[i-1])
                flag = 0
            else:
                move[-1] += distance[i-1]
    command = [move, turn_angle]
    return command


def main_work(frames, model, filters):
    # Wait for a coherent pair of frames: depth and color
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None, None
    depth_frame = filter_depth(depth_frame, filters)
    frame_profile = frames.get_profile()
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    has_pcd, pcd = generate_pcd(color_image, depth_image, frame_profile)
    if not has_pcd:
        return None, None, None, None
    
    # o3d.visualization.draw_geometries([pcd])
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    old_data = dict(coord=points, color=colors)
    old_data, data = center_shift(old_data, apply_z=True)
    voxel = Voxelize(**voxelize_cfg)
    data, idx_unique = voxel(data)
    new_pcd = convert2pcd(old_data['coord'][idx_unique], old_data['color'][idx_unique])
    
    # o3d.visualization.draw_geometries([new_pcd])
    
    ground_found, ground, others, data = find_ground(new_pcd, data)
    if not ground_found:
        return None, None, None, None
    
    # o3d.visualization.draw_geometries([ground])
    
    pred = inference(data, model)
    pred_pcd = prediction2pcd(pred, ground, others)
    
    # o3d.visualization.draw_geometries([pred_pcd])
    
    grid_size = grid_cfg['grid_size']
    
    image = cat_color_depth(color_image, depth_image)
    nodes, ranges = pcd2grid(pred_pcd, grid_size=grid_size)
    path = plan_path(nodes, ranges, grid_size)
    if path is not None:
        command = get_command(path)
        return image, nodes, path, command
    
    return image, nodes, path, None

def main():
    # Load model for detecting trail
    model_path = path_cfg['model_path']
    model = load_model(model_path, model_cfg)
    # Initial figure
    _, axs = plt.subplots(2)
    # Configure depth and color streams
    bag_path = path_cfg['bag_path']
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    # profile = pipeline.start(config)
    # Needed so frames don't get dropped during processing:
    # playback = profile.get_device().as_playback()
    # playback.set_real_time(False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    filters = set_filters()
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            n_frame = frames.frame_number
            if n_frame % camera_cfg['num_frame'] == 0:
                image, nodes, path, command = main_work(frames, model, filters)
                if image is not None and nodes is not None:
                    visualize(axs, image, nodes, path, command)
    finally:
        # Stop streaming
        pass


if __name__ == '__main__':
    main()
