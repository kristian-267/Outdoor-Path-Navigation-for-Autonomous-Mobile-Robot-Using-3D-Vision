import numpy as np
import copy

def find_ground(pcd, data):
    # Find ground using RANSAC
    rest = copy.deepcopy(pcd)
    ground = copy.deepcopy(pcd)
    discarted_segments = []
    ground_found = False
    i = 0
    while not ground_found:
        if i > 3:
            return ground_found, None, None, None
        try:
            segment_models, inliers = rest.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=1000)
        except:
            return ground_found, None, None, None
        dist = np.sqrt(segment_models[0] ** 2 + segment_models[1] ** 2 + segment_models[2] ** 2) + 1e-10
        if len(inliers) >= 5000 and np.abs(segment_models[0] / dist) < 0.5 and np.abs(segment_models[2] / dist) < 0.15:
            ground = rest.select_by_index(inliers)
            for k in data.keys():
                data[k] = data[k][inliers, :]
            ground_found = True
        else:
            discarted_segments.append(rest.select_by_index(inliers))
        rest = rest.select_by_index(inliers, invert=True)
        i += 1
    if len(discarted_segments):
        others = [rest + x for x in discarted_segments][0]
        return ground_found, ground, others, data
    others = rest
    return ground_found, ground, others, data
