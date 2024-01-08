import numpy as np
import open3d as o3d
import os
import glob

result_path = '/Users/kristian/Documents/DTU/Thesis/thesis/data/results'
result_files = glob.glob(f'{result_path}/*.ply')
'''
f = '/Users/kristian/Documents/DTU/Thesis/thesis/data/results/20230315_145631_230.ply'
result = o3d.io.read_point_cloud(f)
o3d.visualization.draw_geometries([result])
'''
# Load point cloud
for f in result_files:
    result = o3d.io.read_point_cloud(f)
    print(f.split('/')[-1])
    o3d.visualization.draw_geometries([result])
