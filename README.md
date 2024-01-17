# Outdoor Path Navigation for Autonomous Mobile Robot Using 3D Vision

## Overview
This repository is dedicated to develop a sophisticated system designed for enabling autonomous mobile robot navigation in outdoor environments using advanced 3D vision technologies and deep learning model. The project integrates various components such as real-time sensor data processing, deep learning model building, training and inference, mapping and path planning algorithms.

## Features
- **Real-time Sensor Data Handling**: Incorporates scripts for handling real-time data from sensors like Intel RealSense cameras.
- **3D Vision and Point Cloud Processing**: Utilizes 3D vision for navigation, incorporating scripts for point cloud generation and processing.
- **Deep Learning Model Integration**: Includes functionality for building and running [SPVCNN](https://arxiv.org/abs/2007.16100) model, with focus on semantic segmentation to detect paths and obstacles.
- **Mapping and Path Planning Algorithms**: Creates grid maps and implements algorithms like Hybrid A* and Reeds-Shepp for efficient pathfinding in complex outdoor environments.
- **Autonomous Navigation**: Combines sensor data processing, deep learning inference, mapping, and path planning for autonomous navigation.

## Contents
- `configs\`: Configuration files for the system.
- `model\`: SPVCNN model used for 3D vision and navigation.
- `references\`: Reference materials and documentation.
- `src\`: Source codes for the implementation.
- `tools\`: Additional tools for the project.

## Requirements
- Ubuntu 18.04
- Python 3.10
- CUDA 12.2

## Installation
- **Copy repository and install required libraries**:
```bash
git clone https://github.com/kristian-267/Outdoor-Path-Navigation-for-Autonomous-Mobile-Robot-Using-3D-Vision.git
cd Outdoor-Path-Navigation-for-Autonomous-Mobile-Robot-Using-3D-Vision
pip install -r requirements
```
- **Install TorchSparse library**:
```bash
pip install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

## Usage
- **Train**:
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
sh tools/train.sh -g ${NUM_GPU} -d dtu_trail -c semseg-spvcnn-all -n ${EXP_NAME}
```
Resume training from checkpoint:
```bash
sh tools/train.sh -g ${NUM_GPU} -d dtu_trail -c semseg-spvcnn-all -n ${EXP_NAME} -r true
```
- **Test**:
```bash
sh scripts/test.sh -g ${NUM_GPU} -d dtu_trail -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
```
- **Navigation**:
Connect Intel RealSense camera mounted on the robot, run:
```bash
export PYTHONPATH=./
python src/navigation/navigation.py
```

## License
The project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements
Credits to Chuansheng Liu. The external resources [Pointcept](https://github.com/Pointcept/Pointcept.git) and [PythonRobotics](https://github.com/kristian-267/PythonRobotics.git) are used in the project.
