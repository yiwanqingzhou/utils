import open3d as o3d
import numpy as np
import math
import copy

from pyquaternion import Quaternion


def matrix2rotation(matrix):
    return matrix[:3, :3]


def matrix2qua(matrix):
    qua = Quaternion(matrix=matrix2rotation(matrix))
    return qua


def matrix2trans(matrix):
    return np.array(matrix[:3, 3:].transpose())[0]


def matrix2pose(matrix):
    qua = matrix2qua(matrix)
    translation = matrix2trans(matrix)
    return np.hstack((translation, qua.elements))


def quatrans2matrix(qua, trans):
    rotaion_matrix = qua.rotation_matrix
    trans_matrix = np.array([trans]).transpose()
    tf_m = np.hstack((rotaion_matrix, trans_matrix))
    m = np.mat([[0, 0, 0, 1]])
    tf_m = np.vstack((tf_m, m))

    return tf_m


def pose2matrix(pose):
    w, x, y, z = pose[3:]
    qua = Quaternion(w=w, x=x, y=y, z=z)
    trans = pose[:3]
    return quatrans2matrix(qua, trans)


def translation2matrix(trans):
    qua = Quaternion(w=1, x=0, y=0, z=0)
    return quatrans2matrix(qua, trans)


def qua2matrix(qua):
    trans = [0, 0, 0]
    return quatrans2matrix(qua, trans)


def load_point_cloud(path):
    """
    Description
    -----------
        read point cloud

    Parameters
    ----------
        path: str

    Returns
    -------
        o3d.geometry.PointCloud
    """
    cloud = o3d.io.read_point_cloud(path)
    return cloud


def get_picked_indices(pcd):
    """
    Description
    -----------
        Pick points of the point cloud in a visualization window

    Parameters
    ----------
        pcd: o3d.geometry.PointCloud

    Returns
    -------
        List
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Resgistration: Pick Corresponding Points")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def sample_mesh_to_cloud(mesh, sample_number):
    """
    Description
    -----------
        process the cabinet mesh model, sample it as point cloud

    Parameters
    ----------
        mesh: o3d.geometry.TriangleMesh
        sample_number: int

    Returns
    -------
        o3d.geometry.PointCloud
    """
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=sample_number)
    pcd.paint_uniform_color([1, 0.706, 0])  # yellow
    return pcd


def draw_registration_result(source, target, transformation):
    """
    Description
    -----------
        Visualize the two point cloud after registration

    Parameters
    ----------
        source: o3d.geometry.PointCloud
        target: o3d.geometry.PointCloud
        transformation: numpy.ndarray[float64[4,4]]

    Returns
    -------
        o3d.geometry.PointCloud
    """
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target], window_name="Registration Visualization"
    )
    return source_temp + target


def get_bounding_box(transform, dimensions):
    """
    Description
    -----------
        get oriented bounding box from transform matrix and dimensions

    Parameters
    ----------
        transform: numpy.matrix
        dimensions: list

    Returns
    -------
        o3d.geometry.OrientedBoundingBox
    """
    rotation = np.array(matrix2rotation(transform))
    center = matrix2trans(transform)
    extent = np.array(dimensions)

    return o3d.geometry.OrientedBoundingBox(center=center, R=rotation, extent=extent)


def get_box_lines_set(transform, dimensions):
    """
    Description
    -----------
        get lines set from transform matrix and dimensions

    Parameters
    ----------
        transform: numpy.matrix
        dimensions: list

    Returns
    -------
        o3d.geometry.LineSet
    """
    box = get_bounding_box(transform, dimensions)
    lines_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)

    return lines_set


def get_cropped_cloud(cloud, transform, dimensions):
    """
    Description
    -----------
        crop cloud according to the transform and dimensions

    Parameters
    ----------
        cloud: o3d.geometry.PointCloud
        transform: numpy.matrix
        dimensions: list

    Returns
    -------
        o3d.geometry.PointCloud
    """
    crop_box = get_bounding_box(transform, dimensions)
    cropped_cloud = cloud.crop(crop_box)
    return cropped_cloud


def calculate_rotate_matrix(points):
    x_axis = points[1] - points[0]
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = points[2] - points[0]
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    return np.vstack((x_axis, y_axis, z_axis)).transpose()


def calculate_tf_matrix(center, points):
    if type(center) is np.matrix:
        center = np.array(center)[0]

    rotation_matrix = calculate_rotate_matrix(points)
    transform_matrix = np.hstack(
        (rotation_matrix, np.array([center]).transpose()))
    transform_matrix = np.vstack((transform_matrix, np.mat([[0, 0, 0, 1]])))

    return transform_matrix


def get_detection_area_matrix_in_cloud(front_bottom, back_bottom, detection_area_dimensions, tf_model_to_cloud):

    rotation = matrix2rotation(tf_model_to_cloud)
    translation = matrix2trans(tf_model_to_cloud)

    origin_in_model = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]

    origin_in_cloud = []
    for point in origin_in_model:
        origin_in_cloud.append(
            np.dot(rotation, point.transpose()) + translation)

    origin_matrix_in_cloud = calculate_tf_matrix(
        origin_in_cloud[0], origin_in_cloud
    )

    l, w, h = detection_area_dimensions

    center_in_model = (front_bottom + back_bottom) / \
        2 + np.array([0, 0, h / 2])

    points_in_model = [
        front_bottom,  # ref pt
        back_bottom,  # pt to x
        front_bottom + np.array([0, -w / 2, 0]),  # pt to y
    ]

    compartment_matrix_in_model = calculate_tf_matrix(
        center_in_model, points_in_model
    )

    tf_compartment_to_detection_area = pose2matrix([0, 0, 0, 1, 0, 0, 0])
    detection_area_matrix_in_model = np.dot(
        compartment_matrix_in_model, tf_compartment_to_detection_area)

    detection_area_matrix_in_cloud = np.dot(
        origin_matrix_in_cloud, detection_area_matrix_in_model)

    return detection_area_matrix_in_cloud


def calibration(source, target, save_path, front_bottom, back_bottom, detection_area_dimensions, icp_threshold):

    while True:
        print('Please press Shift and click the feature points, press Q to finish')
        picked_id_target = get_picked_indices(target)
        print('picked_id_target: ', picked_id_target)

        print('Please press Shift and click the feature points in the same order, press Q to finish')
        picked_id_source = get_picked_indices(source)
        print('picked_id_source: ', picked_id_source)

        if len(picked_id_source) != len(picked_id_target):
            print("The number of corresponding points are not the same, please re-pick")
        else:
            break

    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    p2p = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(
        source, target, o3d.utility.Vector2iVector(corr)
    )

    reg_p2p = o3d.registration.registration_icp(
        source,
        target,
        icp_threshold,
        trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
    )

    transform = reg_p2p.transformation
    print("Transformation Matrix: \n{}\n".format(transform))

    final_cloud = draw_registration_result(
        source, target, transform)

    detection_area_matrix_in_cloud = get_detection_area_matrix_in_cloud(
        front_bottom, back_bottom, detection_area_dimensions, transform)

    detection_area_pose_in_cloud = matrix2pose(
        detection_area_matrix_in_cloud).tolist()

    lines = get_box_lines_set(
        detection_area_matrix_in_cloud, detection_area_dimensions)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=600, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([target, lines, mesh_frame])

    tfed_target = copy.deepcopy(target)
    tfed_target.transform(np.linalg.inv(detection_area_matrix_in_cloud))
    tfed_target.transform(pose2matrix([225, 0, 0, 0, 1, 0, 0]).A)
    o3d.visualization.draw_geometries([tfed_target, mesh_frame])

    o3d.io.write_point_cloud(save_path, tfed_target)


def get_mesh_cloud():
    mesh_box = o3d.geometry.TriangleMesh.create_box(
        width=0.400, height=0.200, depth=0.500)

    mesh_box_cloud = sample_mesh_to_cloud(mesh_box, 100000)
    mesh_box_cloud.transform(pose2matrix([0, 0, 0, 0.707, 0.707, 0, 0]).A)
    mesh_box_cloud.transform(pose2matrix([0, 0, 0, 0.707, 0, 0, 0.707]).A)
    mesh_box_cloud.transform(pose2matrix([0.025, -0.200, -0.200, 1, 0, 0, 0]).A)

    front_box = o3d.geometry.TriangleMesh.create_box(
        width=0.110, height=0.030, depth=0.050)
    front_box_cloud = sample_mesh_to_cloud(front_box, 100000)
    front_box_cloud.transform(pose2matrix([0, 0, 0, 0.707, 0.707, 0, 0]).A)
    front_box_cloud.transform(pose2matrix([0, 0, 0, 0.707, 0, 0, 0.707]).A)
    front_box_cloud.transform(pose2matrix([0, -0.055, -0.030, 1, 0, 0, 0]).A)

    mesh_cloud = mesh_box_cloud + front_box_cloud
    return mesh_cloud


def main():
    load_path = './save.pcd'
    save_path = './tfed_save.pcd'

    source = get_mesh_cloud()
    target = load_point_cloud(load_path)

    front_bottom = np.array([0, 0, -0.200])
    back_bottom = np.array([0.500, 0, -0.200])
    detection_area_dimensions = [0.500, 0.400, 0.200]
    icp_threshold = 0.03

    calibration(source, target, save_path, front_bottom,
                back_bottom, detection_area_dimensions, icp_threshold)


if __name__ == "__main__":
    main()
