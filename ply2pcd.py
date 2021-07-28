from open3d import *

pcd = read_point_cloud("LUCID_HLT003S-001_204700148__20210413103047387_image0.ply")
visualization.draw_geometries([pcd])
write_point_cloud("aa.pcd", pcd)
