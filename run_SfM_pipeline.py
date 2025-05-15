# =========================================
# Incremental SfM Pipeline 
# =========================================

import cv2
import numpy as np
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# =========================================
# Image_loader class to handle image loading and intrinsic matrix handling
# =========================================
class Image_loader():
    def __init__(self, img_dir: str, downscale_factor: float):
        """
        Initializes the image loader by reading the intrinsic matrix (K.txt) and image paths.
        
        Args:
            img_dir (str): Path to the directory containing images and the camera intrinsics file (K.txt).
            downscale_factor (float): Factor by which to downscale both the images and intrinsic matrix.
        """
        # Load the camera intrinsic matrix (3x3) from the 'K.txt' file
        with open(img_dir + '/K.txt') as f:
            self.K = np.array(list((
                map(lambda x: list(map(lambda x: float(x), x.strip().split(' '))),
                    f.read().split('\n')))))

        # Collect image paths from the directory (sorted and filtered for common image formats)
        self.image_list = []
        for image in sorted(os.listdir(img_dir)):
            if image[-4:].lower() == '.jpg' or image[-5:].lower() == '.jpeg' or image[-4:].lower() == '.png':
                self.image_list.append(img_dir + '/' + image)

        # Set current working directory and downscale factor
        self.path = os.getcwd()
        self.factor = downscale_factor

        # Adjust the intrinsic matrix based on the downscale factor
        self.downscale()

    def downscale(self) -> None:
        """
        Downscales the camera intrinsic matrix based on the defined scale factor.
        This ensures compatibility with downscaled images.
        """
        self.K[0, 0] /= self.factor  # fx
        self.K[1, 1] /= self.factor  # fy
        self.K[0, 2] /= self.factor  # cx
        self.K[1, 2] /= self.factor  # cy

    def downscale_image(self, image):
        """
        Downscales the input image using OpenCV's pyrDown function based on the factor.
        
        Args:
            image (np.ndarray): Original image (BGR).
        
        Returns:
            np.ndarray: Downscaled image.
        """
        for _ in range(1, int(self.factor / 2) + 1):
            image = cv2.pyrDown(image)
        return image
# =========================================



# =========================================
# Sfm class to handle Structure from Motion (SfM) operations
# ======================================== 
class Sfm():
    def __init__(self, img_dir: str, downscale_factor: float = 2.0) -> None:
        """
        Initialize the SfM pipeline by loading images and intrinsic matrix.
        
        Args:
            img_dir (str): Path to image dataset.
            downscale_factor (float): Factor to reduce image resolution and adjust intrinsics accordingly.
        """
        self.img_obj = Image_loader(img_dir, downscale_factor)

    def triangulation(self, point_2d_1, point_2d_2, projection_matrix_1, projection_matrix_2) -> tuple:
        """
        Triangulates 3D points from two 2D point correspondences and their projection matrices.

        Args:
            point_2d_1 (np.ndarray): 2D points from first image.
            point_2d_2 (np.ndarray): 2D points from second image.
            projection_matrix_1 (np.ndarray): Projection matrix of first camera.
            projection_matrix_2 (np.ndarray): Projection matrix of second camera.

        Returns:
            tuple: Projection matrices and 3D point cloud in homogeneous coordinates.
        """
        pt_cloud = cv2.triangulatePoints(point_2d_1, point_2d_2, projection_matrix_1.T, projection_matrix_2.T)
        return projection_matrix_1.T, projection_matrix_2.T, (pt_cloud / pt_cloud[3])

    def PnP(self, obj_point, image_point, K, dist_coeff, rot_vector, initial) -> tuple:
        """
        Estimates pose of camera using 3D–2D correspondences (Perspective-n-Point with RANSAC).

        Args:
            obj_point (np.ndarray): 3D points in world.
            image_point (np.ndarray): Corresponding 2D points in image.
            K (np.ndarray): Camera intrinsic matrix.
            dist_coeff (np.ndarray): Distortion coefficients (set to zeros).
            rot_vector (np.ndarray): Rotation vector placeholder.
            initial (int): Flag for first call (1 = reformat inputs).

        Returns:
            tuple: Rotation matrix, translation vector, filtered 2D points, filtered 3D points, updated rotation vector.
        """
        if initial == 1:
            obj_point = obj_point[:, 0, :]
            image_point = image_point.T
            rot_vector = rot_vector.T

        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)

        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            rot_vector = rot_vector[inlier[:, 0]]

        return rot_matrix, tran_vector, image_point, obj_point, rot_vector

    def reprojection_error(self, obj_points, image_points, transform_matrix, K, homogenity) -> tuple:
        """
        Computes the reprojection error between projected 3D points and original 2D image points.

        Args:
            obj_points (np.ndarray): 3D points.
            image_points (np.ndarray): Corresponding 2D points.
            transform_matrix (np.ndarray): Camera [R|t] matrix.
            K (np.ndarray): Camera intrinsics.
            homogenity (int): Flag to convert points from homogeneous if set.

        Returns:
            tuple: Mean reprojection error and possibly updated 3D points.
        """
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        if homogenity == 1:
            obj_points = cv2.convertPointsFromHomogeneous(obj_points.T)

        image_points_calc, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points_calc = np.float32(image_points_calc[:, 0, :])

        total_error = cv2.norm(image_points_calc, np.float32(image_points.T) if homogenity == 1 else np.float32(image_points), cv2.NORM_L2)
        return total_error / len(image_points_calc), obj_points

    def optimal_reprojection_error(self, obj_points) -> np.array:
        """
        Cost function used during bundle adjustment for minimizing reprojection error.

        Args:
            obj_points (np.ndarray): Flattened bundle adjustment variables (pose, intrinsics, 2D/3D points).

        Returns:
            np.ndarray: Flattened reprojection errors.
        """
        transform_matrix = obj_points[0:12].reshape((3, 4))
        K = obj_points[12:21].reshape((3, 3))
        rest = int(len(obj_points[21:]) * 0.4)
        p = obj_points[21:21 + rest].reshape((2, int(rest / 2))).T
        obj_points = obj_points[21 + rest:].reshape((int(len(obj_points[21 + rest:]) / 3), 3))

        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)

        image_points, _ = cv2.projectPoints(obj_points, rot_vector, tran_vector, K, None)
        image_points = image_points[:, 0, :]

        error = [(p[idx] - image_points[idx])**2 for idx in range(len(p))]
        return np.array(error).ravel() / len(p)

    def bundle_adjustment(self, _3d_point, opt, transform_matrix_new, K, r_error) -> tuple:
        """
        Runs non-linear least squares optimization to refine camera parameters and 3D points.

        Args:
            _3d_point (np.ndarray): Initial 3D points.
            opt (np.ndarray): Associated 2D keypoints.
            transform_matrix_new (np.ndarray): Initial camera [R|t] matrix.
            K (np.ndarray): Camera intrinsics.
            r_error (float): Convergence tolerance.

        Returns:
            tuple: Refined 3D points, refined 2D points, refined camera pose.
        """
        opt_variables = np.hstack((transform_matrix_new.ravel(), K.ravel(), opt.ravel(), _3d_point.ravel()))
        values_corrected = least_squares(self.optimal_reprojection_error, opt_variables, gtol=r_error).x

        K = values_corrected[12:21].reshape((3, 3))
        rest = int(len(values_corrected[21:]) * 0.4)

        return (
            values_corrected[21 + rest:].reshape((int(len(values_corrected[21 + rest:]) / 3), 3)),
            values_corrected[21:21 + rest].reshape((2, int(rest / 2))).T,
            values_corrected[0:12].reshape((3, 4))
        )

    def to_ply(self, path, point_cloud, colors) -> None:
        """
        Saves a 3D point cloud with color into a .ply file format for visualization.

        Args:
            path (str): Output folder path.
            point_cloud (np.ndarray): 3D points.
            colors (np.ndarray): Corresponding BGR colors.
        """
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        verts = np.hstack([out_points, out_colors])

        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.linalg.norm(scaled_verts, axis=1)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]

        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        with open(path + '/reconstruction.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')

    def common_points(self, image_points_1, image_points_2, image_points_3) -> tuple:
        """
        Identifies common 2D keypoints across overlapping images to track feature correspondences.

        Args:
            image_points_1 (np.ndarray): Keypoints from image i.
            image_points_2 (np.ndarray): Keypoints from image i+1.
            image_points_3 (np.ndarray): Keypoints from image i+2.

        Returns:
            tuple: (Indices of common points, filtered unmatched sets)
        """
        if image_points_1.size == 0 or image_points_2.size == 0 or image_points_3.size == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )

        cm_points_1 = []
        cm_points_2 = []

        for i in range(image_points_1.shape[0]):
            a = np.where(image_points_2 == image_points_1[i, :])
            if a[0].size != 0:
                cm_points_1.append(i)
                cm_points_2.append(a[0][0])

        mask_array_1 = np.ma.array(image_points_2, mask=False)
        mask_array_1.mask[cm_points_2] = True
        mask_array_1 = mask_array_1.compressed().reshape(-1, 2)

        mask_array_2 = np.ma.array(image_points_3, mask=False)
        mask_array_2.mask[cm_points_2] = True
        mask_array_2 = mask_array_2.compressed().reshape(-1, 2)

        return np.array(cm_points_1, dtype=np.int32), np.array(cm_points_2, dtype=np.int32), mask_array_1, mask_array_2

    def find_features(self, image_0, image_1) -> tuple:
        """
        Detects and matches keypoints between two images using SIFT and Lowe's ratio test.

        Args:
            image_0 (np.ndarray): First image (BGR).
            image_1 (np.ndarray): Second image (BGR).

        Returns:
            tuple: Matched keypoints from image_0 and image_1.
        """
        sift = cv2.SIFT_create()
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        feature = []

        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                feature.append(m)

        return (
            np.float32([key_points_0[m.queryIdx].pt for m in feature]),
            np.float32([key_points_1[m.trainIdx].pt for m in feature])
        )

    # =========================================
    # Main function to run the full SfM pipeline
    # =========================================
    def run_sfm_pipeline(self, enable_bundle_adjustment: boolean = False):
        """
        Executes the full Structure from Motion pipeline:
        - Feature matching
        - Camera pose estimation
        - Triangulation
        - Bundle Adjustment (optional)
        - Saving PLY, pose array, and camera trajectory plots

        Args:
            enable_bundle_adjustment (boolean): If True, performs local bundle adjustment at each step.
        """

        # Create result directory based on dataset name
        res_dir_path = self.img_obj.path + '/output/' + self.img_obj.image_list[0].split('/')[-2]
        if not os.path.exists(res_dir_path):
            os.mkdir(res_dir_path)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        # Initialize data structures
        pose_array = self.img_obj.K.ravel()  # store all projection matrices
        camera_positions = []  # list to track camera centers (trajectory)
        transform_matrix_0 = np.eye(3, 4)  # initial camera pose
        transform_matrix_1 = np.empty((3, 4))

        pose_0 = np.matmul(self.img_obj.K, transform_matrix_0)
        pose_1 = np.empty((3, 4))
        total_points = np.zeros((1, 3))
        total_colors = np.zeros((1, 3))

        # Load and downscale the first two images
        image_0 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[0]))
        image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[1]))

        # Detect SIFT features and matches
        feature_0, feature_1 = self.find_features(image_0, image_1)

        # Save match visualization between first two images
        img_match = cv2.drawMatches(
            image_0,
            [cv2.KeyPoint(pt[0], pt[1], 1) for pt in feature_0],
            image_1,
            [cv2.KeyPoint(pt[0], pt[1], 1) for pt in feature_1],
            [cv2.DMatch(i, i, 0) for i in range(len(feature_0))],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(res_dir_path + "/matches_first_two.png", img_match)

        # Estimate essential matrix and recover relative pose
        essential_matrix, em_mask = cv2.findEssentialMat(feature_0, feature_1, self.img_obj.K, method=cv2.RANSAC, prob=0.999, threshold=0.4)
        if em_mask is not None:
            feature_0 = feature_0[em_mask.ravel() == 1]
            feature_1 = feature_1[em_mask.ravel() == 1]

        _, rot_matrix, tran_matrix, em_mask = cv2.recoverPose(essential_matrix, feature_0, feature_1, self.img_obj.K)
        feature_0 = feature_0[em_mask.ravel() > 0]
        feature_1 = feature_1[em_mask.ravel() > 0]

        # Compose second camera pose using R, t from recoverPose
        transform_matrix_1[:3, :3] = np.matmul(rot_matrix, transform_matrix_0[:3, :3])
        transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], tran_matrix.ravel())

        # Save first two camera positions
        camera_positions.append(np.array([0.0, 0.0, 0.0]))  # origin
        R = transform_matrix_1[:3, :3]
        t = transform_matrix_1[:3, 3].reshape(3, 1)
        camera_center = -R.T @ t
        camera_positions.append(camera_center.ravel())

        pose_1 = np.matmul(self.img_obj.K, transform_matrix_1)

        # Triangulate initial point cloud and compute reprojection error
        feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
        error, points_3d = self.reprojection_error(points_3d, feature_1, transform_matrix_1, self.img_obj.K, homogenity=1)
        print("REPROJECTION ERROR: ", error)

        # Refine 3D–2D correspondences with PnP
        _, _, feature_1, points_3d, _ = self.PnP(points_3d, feature_1, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), feature_0, initial=1)

        total_images = len(self.img_obj.image_list) - 2
        pose_array = np.hstack((pose_array, pose_0.ravel(), pose_1.ravel()))

        threshold = 0.5  # reprojection error threshold

        # Loop through remaining images
        for i in tqdm(range(total_images)):
            print(self.img_obj.image_list[i + 2])
            image_2 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
            features_cur, features_2 = self.find_features(image_1, image_2)

            # Triangulate from previous step if not first iteration
            if i != 0:
                feature_0, feature_1, points_3d = self.triangulation(pose_0, pose_1, feature_0, feature_1)
                feature_1 = feature_1.T
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)[:, 0, :]

            # Find common points across three frames
            cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = self.common_points(feature_1, features_cur, features_2)
            cm_points_2 = features_2[cm_points_1]
            cm_points_cur = features_cur[cm_points_1]

            # If not enough matches for PnP, skip frame
            if len(cm_points_0) < 4:
                print(f"Skipping frame {i+2} due to insufficient points for PnP ({len(cm_points_0)})")
                image_0 = np.copy(image_1)
                image_1 = self.img_obj.downscale_image(cv2.imread(self.img_obj.image_list[i + 2]))
                feature_0 = np.copy(features_cur)
                feature_1 = np.copy(features_2)
                pose_0 = np.copy(pose_1)
                continue

            # Estimate new pose using PnP
            rot_matrix, tran_matrix, cm_points_2, points_3d, cm_points_cur = self.PnP(
                points_3d[cm_points_0], cm_points_2, self.img_obj.K, np.zeros((5, 1), dtype=np.float32), cm_points_cur, initial=0)

            transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
            pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)

            # Save new camera center
            R = transform_matrix_1[:3, :3]
            t = transform_matrix_1[:3, 3].reshape(3, 1)
            camera_center = -R.T @ t
            camera_positions.append(camera_center.ravel())

            error, points_3d = self.reprojection_error(points_3d, cm_points_2, transform_matrix_1, self.img_obj.K, homogenity=0)

            # Triangulate new points
            cm_mask_0, cm_mask_1, points_3d = self.triangulation(pose_1, pose_2, cm_mask_0, cm_mask_1)
            error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity=1)
            print("Reprojection Error: ", error)
            pose_array = np.hstack((pose_array, pose_2.ravel()))

            # Optional: Local bundle adjustment
            if enable_bundle_adjustment:
                points_3d, cm_mask_1, transform_matrix_1 = self.bundle_adjustment(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, threshold)
                pose_2 = np.matmul(self.img_obj.K, transform_matrix_1)
                error, points_3d = self.reprojection_error(points_3d, cm_mask_1, transform_matrix_1, self.img_obj.K, homogenity=0)
                print("Bundle Adjusted error: ", error)

            # Accumulate 3D points and color
            if enable_bundle_adjustment:
                total_points = np.vstack((total_points, points_3d))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left])
            else:
                total_points = np.vstack((total_points, points_3d[:, 0, :]))
                points_left = np.array(cm_mask_1, dtype=np.int32)
                color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
            total_colors = np.vstack((total_colors, color_vector))

            # Update previous frame variables
            transform_matrix_0 = np.copy(transform_matrix_1)
            pose_0 = np.copy(pose_1)
            plt.scatter(i, error)
            plt.savefig(res_dir_path + '/reprojection_error_plot.png')
            plt.pause(0.05)

            image_0 = np.copy(image_1)
            image_1 = np.copy(image_2)
            feature_0 = np.copy(features_cur)
            feature_1 = np.copy(features_2)
            pose_1 = np.copy(pose_2)

            # Show image
            cv2.imshow(self.img_obj.image_list[0].split('/')[-2], image_2)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        # =========================================
        # All images are parsed, the loop is done
        # =========================================

        cv2.destroyAllWindows()
        # Save reconstructed point cloud and camera pose array
        print("Saved data .ply file")
        self.to_ply(res_dir_path, total_points, total_colors)
        np.savetxt(res_dir_path + '/pose_array.csv', pose_array, delimiter='\n')
        # Save Camera Trajectory Plots 
        camera_positions = np.array(camera_positions)
        # Save 2D X-Z trajectory plot
        plt.figure()
        plt.plot(camera_positions[:, 0], camera_positions[:, 2], marker='o')
        plt.title("Camera Trajectory (XZ Plane)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.grid()
        plt.axis("equal")
        plt.savefig(res_dir_path + "/camera_trajectory.png")
        plt.close()
        # Save 3D trajectory plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], marker='o')
        ax.set_title("Camera Trajectory (3D)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.savefig(res_dir_path + "/camera_trajectory_3D.png")
        plt.close()



###########################################
# =========================================
# =========================================
#### MAIN FUNCTION TO RUN #################
# =========================================
# =========================================
###########################################
if __name__ == "__main__":

    # Example usage on terminal:
    # python main.py dataset_name FOR EXAMPLE python main.py status
    # Dataset saved in res folder

    # Check if a dataset name is provided as a command line argument
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:], 1):
            sfm = Sfm("Datasets/" + arg)
            sfm.run_sfm_pipeline(enable_bundle_adjustment=True) 
    
    # If no dataset name is provided, use the default dataset
    else:
        print("No dataset name provided so using the default dataset")
        sfm = Sfm("Datasets/statue")
        sfm.run_sfm_pipeline(enable_bundle_adjustment=True)
