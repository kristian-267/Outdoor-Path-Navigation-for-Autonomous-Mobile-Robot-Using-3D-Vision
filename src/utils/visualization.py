import matplotlib.pyplot as plt
import numpy as np

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.libs.HybridAStar.car import plot_car

def visualize(axs, image, nodes, path, command):
    axs[0].cla()
    try:
        axs[0].imshow(image)
    except:
        pass
    axs[1].cla()
    try:
        if len(nodes['obstacles']):
            axs[1].scatter(np.array(nodes['obstacles'][0]), np.array(nodes['obstacles'][1]), c='black', s=10)
        if len(nodes['others']):
            axs[1].scatter(np.array(nodes['others'][0]), np.array(nodes['others'][1]), c='b', s=10)
        if len(nodes['trail']):
            axs[1].scatter(np.array(nodes['trail'][0]), np.array(nodes['trail'][1]), c='r', s=10)
        if len(nodes['pavement']):
            axs[1].scatter(np.array(nodes['pavement'][0]), np.array(nodes['pavement'][1]), c='y', s=10)
        if len(nodes['motorway']):
            axs[1].scatter(np.array(nodes['motorway'][0]), np.array(nodes['motorway'][1]), c='lime', s=10)
        if path is not None:
            x = np.array(path.x_list)
            y = np.array(path.y_list)
            axs[1].plot(x, y, "-k", label="Hybrid A* path")
            plot_car(x[0], y[0], path.yaw_list[0])
            for i in range(len(command[0])):
                if i > 3:
                    break
                if command[1][i] != 0:
                    axs[1].text(0.01, 0.13 - 0.04 * i, 'Step {}: Turn {} degrees and move {}m'.format(i + 1, round(command[1][i]), round(command[0][i], 1)), fontsize=10, transform=plt.gcf().transFigure)
                else:
                    axs[1].text(0.01, 0.13 - 0.04 * i, 'Step {}: Move straight {}m'.format(i + 1, round(command[0][i], 1)), fontsize=10, transform=plt.gcf().transFigure)
            plt.subplots_adjust(bottom=0.2)
        axs[1].grid(True)
    except:
        pass
    plt.pause(0.000001)
