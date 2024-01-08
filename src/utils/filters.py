import pyrealsense2 as rs

def set_filters():
    # Processing blocks
    hdr_merge = rs.hdr_merge()
    
    depth_to_disparity = rs.disparity_transform(True)
    
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 4)
    
    temporal = rs.temporal_filter(smooth_alpha=0.4, smooth_delta=20, persistence_control=3)
    
    disparity_to_depth = rs.disparity_transform(False)
    
    threshold = rs.threshold_filter(0.6, 8.0)
    
    filters = dict(
        hdr_merge=hdr_merge,
        depth_to_disparity=depth_to_disparity,
        spatial=spatial,
        temporal=temporal,
        disparity_to_depth=disparity_to_depth,
        threshold=threshold
    )
    return filters

def filter_depth(depth_frame, filters):
    for k in filters.keys():
        depth_frame = filters[k].process(depth_frame)
    return depth_frame