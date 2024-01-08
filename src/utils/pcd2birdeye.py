import numpy as np

# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
  """ Scales an array of values from specified min, max range to 0-255
      Optionally specify the data type of the output (default is uint8)
  """
  return (((a - min) / float(max - min)) * 255).astype(dtype)

def get_range(coord):
    x_min = np.ceil(np.min(coord[:, 0]) * 10) / 10
    x_max = int(np.max(coord[:, 0]) * 10) / 10
    y_max = np.max(coord[:, 1])
    z_max = int(np.max(coord[:, 2]) * 10) / 10
    side_range = (x_min, x_max)
    fwd_range = (0., z_max)
    height_range = (-0.3, y_max)
    return side_range, fwd_range, height_range
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_birdseye(pcd, res):
    """ Creates an 2D birds eye view representation of the point cloud data.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    side_range, fwd_range, height_range = get_range(points)
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 2]
    y_points = -points[:, 0]
    z_points = points[:, 1]
    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points < -side_range[0]), (y_points > -side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    colors = colors[indices]
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR
    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))
    x_img -= 1
    y_img -= 1
    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])
    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(colors,
                                min=0.,
                                max=1.)
    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.full((y_max, x_max, 3), 255, dtype=np.uint8)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img, :] = pixel_values
    return im, side_range, fwd_range
