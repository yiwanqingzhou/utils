from open3d import *

pcd = read_point_cloud("LX311f-ASM-2019.3.8-.xyz")
write_point_cloud("LX311f-ASM-2019.3.8-.pcd", pcd)
