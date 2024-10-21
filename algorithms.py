import cv2
import numpy as np
import time


def measure_performance(matcher, image1, image2):
    """
    Measures the performance of the keypoint matching process between two images.

    :param matcher: An instance of a matcher class (e.g., SIFTMatcher, ORBMatcher).
    :param image1: The first image for matching.
    :param image2: The second image for matching.
    :return: A tuple containing the number of good matches and the execution time.
    """
    # Record the start time
    start_time = time.time()

    # Find keypoints and descriptors for both images
    keypoints1, descriptors1 = matcher.find_keypoints_and_descriptors(image1)
    keypoints2, descriptors2 = matcher.find_keypoints_and_descriptors(image2)

    # Match the descriptors between the two images
    good_matches = matcher.match(descriptors1, descriptors2)

    # Record the end time
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    num_matches = len(good_matches)

    return num_matches, execution_time


class SIFTMatcher:
    def __init__(self, ratio_thresh=0.6):
        """
        Initializes the SIFT matcher with a given ratio threshold for filtering matches.

        :param ratio_thresh: The threshold for the ratio test used to filter matches.
        """
        self.ratio_thresh = ratio_thresh
        self.sift = cv2.SIFT_create()  # Create a SIFT detector object

    def find_keypoints_and_descriptors(self, image):
        """
        Finds keypoints and their descriptors in a given image using SIFT.

        :param image: The image in which to find keypoints.
        :return: A tuple of keypoints and descriptors.
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        keypoints, descriptors = self.sift.detectAndCompute(grayscale, None)  # Detect keypoints and compute descriptors
        return keypoints, descriptors

    def match(self, descriptors1, descriptors2):
        """
        Matches descriptors from two images using the KNN method.

        :param descriptors1: Descriptors from the first image.
        :param descriptors2: Descriptors from the second image.
        :return: A list of good matches based on the ratio test.
        """
        bf = cv2.BFMatcher(cv2.NORM_L2)  # Create a brute force matcher object
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # Find the two best matches for each descriptor
        # Apply the ratio test to filter out poor matches
        good = [m for m, n in matches if m.distance < self.ratio_thresh * n.distance]
        return good


class RANSACFilter:
    def __init__(self, reprojection_thresh=2.0):
        """
        Initializes the RANSAC filter with a reprojection threshold.

        :param reprojection_thresh: The threshold for RANSAC to consider inliers.
        """
        self.reprojection_thresh = reprojection_thresh

    def apply(self, keypoints1, keypoints2, matches):
        """
        Applies RANSAC to filter matches based on geometric consistency.

        :param keypoints1: Keypoints from the first image.
        :param keypoints2: Keypoints from the second image.
        :param matches: List of matches to filter.
        :return: A homography matrix and a mask indicating inliers.
        """
        if not matches:
            return None, None

        # Get matched points from keypoints
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute the homography matrix using RANSAC
        M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.reprojection_thresh)
        return M, mask

    def check_geometric_consistency(self, keypoints1, keypoints2, good_matches, M, threshold=5):
        """
        Checks the geometric consistency of matches using the homography matrix.

        :param keypoints1: Keypoints from the first image.
        :param keypoints2: Keypoints from the second image.
        :param good_matches: List of good matches.
        :param M: The homography matrix.
        :param threshold: The distance threshold for inliers.
        :return: The count of inliers and a mask indicating inliers.
        """
        # Get matched points for inlier checking
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Project points using the homography matrix
        pts2_proj = cv2.perspectiveTransform(pts1, M)
        # Calculate distances between projected and actual points
        distances = np.sqrt(np.sum((pts2 - pts2_proj) ** 2, axis=2))

        # Create a mask for inliers based on the distance threshold
        inlier_mask = distances < threshold
        inlier_count = np.sum(inlier_mask)

        return inlier_count, inlier_mask


class ORBMatcher:
    def __init__(self, number_of_points=1000, fast_threshold=20):
        """
        Initializes the ORB matcher with a specified number of keypoints and FAST threshold.

        :param number_of_points: The maximum number of features to retain.
        :param fast_threshold: The threshold for the FAST feature detector.
        """
        self.orb = cv2.ORB_create(nfeatures=number_of_points, fastThreshold=fast_threshold)

    def find_keypoints_and_descriptors(self, image):
        """
        Finds keypoints and their descriptors in a given image using ORB.

        :param image: The image in which to find keypoints.
        :return: A tuple of keypoints and descriptors.
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        keypoints, descriptors = self.orb.detectAndCompute(grayscale, None)  # Detect keypoints and compute descriptors
        return keypoints, descriptors

    def match(self, descriptors1, descriptors2):
        """
        Matches descriptors from two images using a brute force matcher with Hamming distance.

        :param descriptors1: Descriptors from the first image.
        :param descriptors2: Descriptors from the second image.
        :return: A sorted list of matches based on distance.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Create a brute force matcher using Hamming distance
        matches = bf.match(descriptors1, descriptors2)  # Match descriptors
        matches = sorted(matches, key=lambda x: x.distance)  # Sort matches based on distance
        return matches


def visualize_inliers(image1, image2, keypoints1, keypoints2, good_matches, inlier_mask):
    """
    Visualizes inlier matches between two images.

    :param image1: The first image to visualize.
    :param image2: The second image to visualize.
    :param keypoints1: Keypoints from the first image.
    :param keypoints2: Keypoints from the second image.
    :param good_matches: List of good matches to visualize.
    :param inlier_mask: Mask indicating which matches are inliers.
    """
    # Draw keypoints for both images
    image1_inliers = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2_inliers = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Draw circles around the inlier matches
    for i, match in enumerate(good_matches):
        if inlier_mask[i]:
            pt1 = tuple(map(int, keypoints1[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints2[match.trainIdx].pt))
            cv2.circle(image1_inliers, pt1, 5, (0, 255, 255), -1)
            cv2.circle(image2_inliers, pt2, 5, (0, 255, 255), -1)

    # Display the images with inliers highlighted
    cv2.imshow('Inliers Image 1', image1_inliers)
    cv2.imshow('Inliers Image 2', image2_inliers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
