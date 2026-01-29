from argparse import ArgumentParser

import open3d as o3d
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('ply_files', nargs='+')
    args = parser.parse_args()

    return args

# Read the .ply file
def main():
    args = parse_args()
    pcds = [
        o3d.io.read_point_cloud(path)
        for path in args.ply_files
    ]
    # Visualize the point cloud (optional)
    o3d.visualization.draw_geometries(pcds)


if __name__ == '__main__':
    main()
