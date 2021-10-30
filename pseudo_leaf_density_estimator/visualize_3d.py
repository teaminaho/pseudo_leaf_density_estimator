#!/usr/bin/env python3

import numpy as np
import open3d as o3d


def make_pcd(xyz, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)
    return pcd


xyz = np.random.random((100, 3))
pcd_r = make_pcd(xyz, color=(1., 0., 0.))

xyz = np.random.random((100, 3))
pcd_g = make_pcd(xyz, color=(0., 1., 0.))

o3d.visualization.draw_geometries([pcd_r + pcd_g])
