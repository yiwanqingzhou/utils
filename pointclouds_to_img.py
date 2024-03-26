import numpy as np
import open3d as o3d
import cv2 as cv
import os, os.path as osp

import sys

def extract_image_and_points(cloud, cloud_height, cloud_width, flip_rgb=True):
    points = np.asarray(cloud.points)
    color = np.asarray(cloud.colors)

    img = color.reshape((cloud_height, cloud_width, 3))
    if flip_rgb:
        img = img[:, :, ::-1].copy()
    points = points.reshape((cloud_height, cloud_width, 3))
    return img, points


def read_point_cloud(pcd_file, point_multiplier=1.0):
    if not osp.exists(pcd_file):
        return False, None

    cloud = o3d.read_point_cloud_with_nan(pcd_file)
    if cloud.is_empty():
        return False, None

    if point_multiplier is not None and point_multiplier != 1.0:
        cloud.points = o3d.Vector3dVector(np.asarray(cloud.points) * point_multiplier)
    return True, cloud


if __name__ == '__main__':

    COLOR_HEIGHT = 540
    COLOR_WIDTH = 960
    pcd_name = sys.argv[1]
    pcd_path = pcd_name + '.pcd'
    rt, pcd = read_point_cloud(pcd_path)
    if not rt:
        print("failed to read the points")
        sys.exit(0)

    img, cloud_points = extract_image_and_points(pcd, COLOR_HEIGHT, COLOR_WIDTH)
    img = (img * 255).astype(np.uint8)
    cv.imwrite(pcd_name + ".png", img)

