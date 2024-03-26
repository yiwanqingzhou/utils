import open3d as o3d
import sys

path = str(sys.argv[1])
pcd = o3d.io.read_point_cloud(path)

new_path = path[:-3] + "ply"
print(path, " -> ", new_path)
o3d.io.write_point_cloud(new_path, pcd)
