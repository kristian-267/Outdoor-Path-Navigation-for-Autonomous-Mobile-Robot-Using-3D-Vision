import torch
import torch.nn.functional as F
import collections
import numpy as np
import open3d as o3d
import os
import copy
import time
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.models import build_model
from configs.navigation.navigation import model_cfg, voxelize_cfg, path_cfg
from src.dataset.data_prepare import normalize_color, center_shift, to_tensor
from src.utils.voxelize import Voxelize


def data_load(data_path):
    data = torch.load(data_path)
    coord = data["coord"]
    color = data["color"]
    segment = data["semantic_gt"].reshape(-1)
    data = dict(coord=coord, color=color, segment=segment)
    return data


device = torch.device('cpu')
model = build_model(model_cfg)
model.to(device)

state_dict = torch.load(path_cfg['model_path'], map_location=device)
new_state_dict = collections.OrderedDict()
for name, value in state_dict.items():
    if name.startswith("module."):
        name = name[7:]  # module.xxx.xxx -> xxx.xxx
    new_state_dict[name] = value
model.load_state_dict(new_state_dict, strict=True)
model.eval()

file_names = ['20230315_145631_5900', '20230315_145631_6395', '20230315_145631_8765', '20230315_145631_15460',
              '20230315_145631_16070', '20230619_131140_300', '20230619_131140_1580', '20230619_131140_1640',
              '20230619_131140_2840', '20230619_132333_125', '20230619_132333_2550', '20230619_132333_2600']

inf_times = []

for file_name in file_names:
    ground_path = '/mnt/Documents/DTU/Thesis/report/presented_data/extra_test/ground/{}.ply'.format(file_name)
    label_path = '/mnt/Documents/DTU/Thesis/report/presented_data/extra_test/ground_labels/{}.ply'.format(file_name)
    pcd_bw_path = '/mnt/Documents/DTU/Thesis/report/presented_data/extra_test/pcd_bw/{}.ply'.format(file_name)

    ground = o3d.io.read_point_cloud(ground_path)
    points = np.asarray(ground.points)
    colors = np.asarray(ground.colors) * 255

    ground_label = o3d.io.read_point_cloud(label_path)
    label_colors = np.asarray(ground_label.colors) * 255

    label = np.where((label_colors[:, 0] == 1) & (label_colors[:, 1] == 1) & (label_colors[:, 2] == 1), 1,
                    np.where((label_colors[:, 0] == 2) & (label_colors[:, 1] == 2) & (label_colors[:, 2] == 2), 2,
                            np.where((label_colors[:, 0] == 3) & (label_colors[:, 1] == 3) & (label_colors[:, 2] == 3), 3,
                                    0)))
    label = label[:, np.newaxis]

    pcd_bw = o3d.io.read_point_cloud(pcd_bw_path)
    bw_points = np.asarray(pcd_bw.points)
    bw_colors = np.asarray(pcd_bw.colors)
    mask = np.where((bw_colors[:, 0] == 0) & (bw_colors[:, 1] == 0) & (bw_colors[:, 2] == 0), 1, 0)
    bw_points = bw_points[mask==1]
    bw_colors = bw_colors[mask==1]

    bw_data = dict(coord=copy.deepcopy(bw_points), color=copy.deepcopy(bw_colors))
    data = dict(coord=copy.deepcopy(points), color=copy.deepcopy(colors), segment=copy.deepcopy(label))

    label = data['segment']
    _, data = center_shift(data, apply_z=True)
    _, bw_data = center_shift(bw_data, apply_z=True)

    voxel = Voxelize(voxel_size=0.05)
    data, idx = voxel(data)
    bw_data, bw_idx = voxel(bw_data)

    data = normalize_color(data)
    _, data = center_shift(data, apply_z=False)
    data = to_tensor(data)

    points = points[idx]
    label = label[idx]

    bw_points = bw_points[bw_idx]
    bw_colors = bw_colors[bw_idx]

    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cpu()
            
    start_time = time.time()
    with torch.no_grad():
        output = model(data)
        pred = output["seg_logits"]
        pred = F.softmax(pred, -1)
    inf_time = time.time() - start_time
    inf_times.append(inf_time)

    pred = pred.max(1)[1].data.numpy()
    label = label.reshape(-1)
    accuracy = np.sum(np.equal(pred, label)) / pred.shape[0]

    # print('loss:', np.around(loss.numpy(), 4))
    print('accuracy:', np.around(accuracy, 4))

    target_colors = np.zeros([len(pred), 3])
    target_colors[label == 0] = [0, 0, 1]
    target_colors[label == 1] = [1, 0, 0]
    target_colors[label == 2] = [1, 1, 0]
    target_colors[label == 3] = [0, 1, 0]

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(points)
    target_pcd.colors = o3d.utility.Vector3dVector(target_colors)

    pred_colors = np.zeros([len(pred), 3])
    pred_colors[pred == 0] = [0, 0, 1]
    pred_colors[pred == 1] = [1, 0, 0]
    pred_colors[pred == 2] = [1, 1, 0]
    pred_colors[pred == 3] = [0, 1, 0]

    pred_pcd = o3d.geometry.PointCloud()
    pred_pcd.points = o3d.utility.Vector3dVector(points)
    pred_pcd.colors = o3d.utility.Vector3dVector(pred_colors)

    obstacle = o3d.geometry.PointCloud()
    obstacle.points = o3d.utility.Vector3dVector(bw_points)
    obstacle.colors = o3d.utility.Vector3dVector(bw_colors)

    whole_pcd = pred_pcd + obstacle
    '''
    o3d.visualization.draw_geometries([ground],
                                    zoom=0.5,
                                    front=[0.0, -1.0, -1.0],
                                    lookat=[0.0, -3.0, 0.0],
                                    up=[0.0, -1.0, -0.0])
    o3d.visualization.draw_geometries([target_pcd],
                                    zoom=0.5,
                                    front=[0.0, -1.0, -1.0],
                                    lookat=[0.0, -3.0, 0.0],
                                    up=[0.0, -1.0, -0.0])
    o3d.visualization.draw_geometries([pred_pcd],
                                    zoom=0.5,
                                    front=[0.0, -1.0, -1.0],
                                    lookat=[0.0, -3.0, 0.0],
                                    up=[0.0, -1.0, -0.0])
    
    o3d.visualization.draw_geometries([obstacle],
                                    zoom=0.5,
                                    front=[0.0, -1.0, -1.0],
                                    lookat=[0.0, -3.0, 0.0],
                                    up=[0.0, -1.0, -0.0])
    o3d.visualization.draw_geometries([whole_pcd],
                                    zoom=0.5,
                                    front=[0.0, -1.0, -1.0],
                                    lookat=[0.0, -3.0, 0.0],
                                    up=[0.0, -1.0, -0.0])
    '''

    # o3d.io.write_point_cloud(os.path.join('/mnt/Documents/DTU/Thesis/report/presented_data/whole_pcd', file_name + '.ply'), whole_pcd)

print(np.mean(np.array(inf_times)))

print('end')
