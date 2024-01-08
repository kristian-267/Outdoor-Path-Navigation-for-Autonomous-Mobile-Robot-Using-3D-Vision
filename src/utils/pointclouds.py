import numpy as np
import open3d as o3d
from open3d.geometry import Image, RGBDImage, PointCloud
from open3d.camera import PinholeCameraIntrinsic

def generate_pcd(color_image, depth_image, frame_profile):
    has_pcd = False
    if np.max(depth_image) == 0:
       return has_pcd, None
    img_depth = Image(depth_image)
    img_color = Image(color_image)
    rgbd = RGBDImage.create_from_color_and_depth(img_color, img_depth, depth_scale=1000.0, depth_trunc=8.0, convert_rgb_to_intensity=False)
    intrinsics = frame_profile.as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
    pcd = PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    has_pcd = True
    return has_pcd, pcd

def convert2pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def prediction2pcd(pred, ground, others):
    pred_colors = np.zeros([len(pred), 3])
    pred_colors[pred == 0] = [0, 0, 1]
    pred_colors[pred == 1] = [1, 0, 0]
    pred_colors[pred == 2] = [1, 1, 0]
    pred_colors[pred == 3] = [0, 1, 0]
    ground.colors = o3d.utility.Vector3dVector(pred_colors)
    others.paint_uniform_color([0, 0, 0])
    pcd = ground + others
    return pcd
