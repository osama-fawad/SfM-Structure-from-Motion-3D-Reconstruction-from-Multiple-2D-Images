import cv2
import numpy as np
import glob
import os

def calibrate_camera_from_chessboard(
    image_folder,
    pattern_size=(9, 7),
    square_size=1.0,
    output_folder="calibration_output"
):
    """
    Calibrates a single pinhole camera from a set of chessboard images.
    
    Args:
        image_folder (str): Folder containing input calibration images.
        pattern_size (tuple): Number of inner corners per chessboard row and column (cols, rows).
        square_size (float): Real-world square size (in any unit, e.g., mm or cm).
                             Doesn't affect intrinsic K, but affects scale of 3D results.
        output_folder (str): Directory where all outputs (K.txt, images, error.txt) are saved.

    Returns:
        K (np.ndarray): Camera intrinsic matrix.
        dist (np.ndarray): Distortion coefficients.
        mean_error (float): Mean reprojection error across all images.
    """

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Prepare object points in real world space (Z = 0 plane)
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # multiply by real-world square size

    objpoints = []  # 3D world points
    imgpoints = []  # 2D image points

    # Load all chessboard images
    images = glob.glob(os.path.join(image_folder, "*.jpg")) + \
             glob.glob(os.path.join(image_folder, "*.png")) + \
             glob.glob(os.path.join(image_folder, "*.jpeg"))

    print(f"Found {len(images)} calibration images.")

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corner locations
            corners2 = cv2.cornerSubPix(
                gray, corners, winSize=(11,11), zeroZone=(-1,-1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Save visualized corners
            # vis = cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            vis_path = os.path.join(output_folder, f"corners_{idx:02}.jpg")
            # cv2.imwrite(vis_path, vis)

            # Clone image for drawing
            vis = img.copy()

            # Draw each corner as a bold cross
            for pt in corners2:
                pt = tuple(int(x) for x in pt.ravel())
                cv2.drawMarker(vis, pt, color=(0, 100, 255), markerType=cv2.MARKER_CROSS,
                            markerSize=18, thickness=15, line_type=cv2.LINE_AA)

            # Optionally: draw lines between consecutive corners
            for i in range(len(corners2) - 1):
                pt1 = tuple(int(x) for x in corners2[i].ravel())
                pt2 = tuple(int(x) for x in corners2[i + 1].ravel())
                cv2.line(vis, pt1, pt2, color=(255, 100, 0), thickness=12, lineType=cv2.LINE_AA)

            # Save the bold visualization
            cv2.imwrite(vis_path, vis)

        else:
            print(f"Warning - Chessboard NOT detected in: {img_path}")

    if len(objpoints) < 3:
        raise RuntimeError("Not enough valid calibration images with detected corners.")

    # Run camera calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Print intrinsics
    print("\n=== Camera Intrinsics ===")
    print("K Matrix:\n", K)
    print("Distortion Coefficients:\n", dist.ravel())

    # Save K to .npz and .txt
    np.savez(os.path.join(output_folder, "intrinsics.npz"), K=K, dist=dist)

    k_txt_path = os.path.join(output_folder, "K.txt")
    with open(k_txt_path, "w") as f:
        for row in K:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    print(f"K matrix saved to {k_txt_path}")

    # Compute reprojection error
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(imgpoints_proj)

    mean_error = np.sqrt(total_error / total_points)
    print(f"\nMean Reprojection Error: {mean_error:.6f} pixels")

    # Save error
    error_path = os.path.join(output_folder, "reprojection_error.txt")
    with open(error_path, "w") as f:
        f.write(f"Mean Reprojection Error: {mean_error:.6f} pixels\n")
    print(f"\nError saved to {error_path}")

    return K, dist, mean_error


if __name__ == "__main__":

    # use calibrate function
    image_folder = "Datasets/custom_dataset_laptop/calibration_images"
    pattern_size = (9, 7)  # Number of inner corners per chessboard row and column
    square_size = 1.0  # Real-world square size (in any unit, e.g., mm or cm)
    output_folder = "output/calibration_output"  # Directory to save outputs
    K, dist, mean_error = calibrate_camera_from_chessboard(
        image_folder,
        pattern_size,
        square_size,
        output_folder
    )

    # print(f"Calibration completed. K: {K}, Distortion: {dist}, Mean Error: {mean_error}")
    
### OUTPUT ###
"""
Found 10 calibration images.

=== Camera Intrinsics ===
K Matrix:
 [[3.13007799e+03 0.00000000e+00 1.38925889e+03]
 [0.00000000e+00 3.07648643e+03 1.95541320e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Distortion Coefficients:
 [ 0.12562487 -0.11168789 -0.0078617  -0.01553253 -0.19819287]
K matrix saved to output/calibration_output/K.txt

Mean Reprojection Error: 1.657573 pixels

Error saved to output/calibration_output/reprojection_error.txt
"""