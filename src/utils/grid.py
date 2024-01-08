import numpy as np
import open3d as o3d

def pcd2grid(pcd, grid_size):
    points = np.asarray(pcd.points)
    
    x_min = np.ceil(np.min(points[:, 0]) * 10) / 10
    z_min = int(np.min(points[:, 2]) * 10) / 10
    x_max = int(np.max(points[:, 0]) * 10) / 10
    y_max = np.max(points[:, 1])
    z_max = int(np.max(points[:, 2]) * 10) / 10

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(x_min, 0.0, z_min), max_bound=(x_max, y_max, z_max))
    new_pcd = pcd.crop(bbox)
    # Calculate number of grid cells in each direction
    x_cells = int(np.ceil((x_max - x_min) / grid_size))
    z_cells = int(np.ceil((z_max - z_min) / grid_size))
    # Create empty list to hold the grid cells
    grid_cells = [[] for i in range(x_cells * z_cells)]
    grid_x = np.linspace(x_min + 0.5 * grid_size, x_max - 0.5 * grid_size, x_cells)
    grid_z = np.linspace(z_min + 0.5 * grid_size, z_max - 0.5 * grid_size, z_cells)
    
    obstacles = [[], []]
    trail = [[], []]
    pavement = [[], []]
    motorway = [[], []]
    others = [[], []]
    for x in range(x_cells):
        for z in range(z_cells):
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(x_min + x * grid_size, 0, z_min + z * grid_size),
                max_bound=(x_min + (x + 1) * grid_size, y_max, z_min + (z + 1) * grid_size))
            grid = new_pcd.crop(bbox)
            grid_cells[x + z * x_cells] = grid
            if len(grid.points) <= 0:
                continue
            pos = [np.around(grid_x[x], 2), np.around(grid_z[z], 2)]
            grid_colors = np.asarray(grid.colors)
            if np.sum((grid_colors[:, 0] == 0) & (grid_colors[:, 1] == 0) & (grid_colors[:, 2] == 0)):
                obstacles[0].append(pos[0])
                obstacles[1].append(pos[1])
            elif np.sum((grid_colors[:, 0] == 0) & (grid_colors[:, 1] == 1) & (grid_colors[:, 2] == 0)) / len(grid.points) >= 0.5:
                motorway[0].append(pos[0])
                motorway[1].append(pos[1])
            elif np.sum((grid_colors[:, 0] == 1) & (grid_colors[:, 1] == 0) & (grid_colors[:, 2] == 0)) / len(grid.points) >= 0.1:
                trail[0].append(pos[0])
                trail[1].append(pos[1])
            elif np.sum((grid_colors[:, 0] == 1) & (grid_colors[:, 1] == 1) & (grid_colors[:, 2] == 0)) / len(grid.points) >= 0.1:
                pavement[0].append(pos[0])
                pavement[1].append(pos[1])
            else:
                others[0].append(pos[0])
                others[1].append(pos[1])
    
    nodes = dict(obstacles=obstacles, others=others, trail=trail, pavement=pavement, motorway=motorway)
    ranges = dict(x_min=x_min, x_max=x_max, y_min=0.0, y_max=z_max)
    return nodes, ranges
