import numpy as np
import open3d as o3d
import os
import glob

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

result_path = '/results'
result_files = glob.glob(f'{result_path}/*.ply')

# Load point clouds
for f in result_files:
    result = o3d.io.read_point_cloud(f)
    print(f.split('/')[-1])
    o3d.visualization.draw_geometries([result])
