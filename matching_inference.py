import logging
import os

from matplotlib import pyplot as plt

from algorithms import *

def draw_matches(image1, keypoints1, image2, keypoints2, matches, title='Matches', scale_percent=50):
    '''

   Draw matches between two images using keypoints and descriptors.

    :param image1: First image (numpy array) from which keypoints are detected.
    :param keypoints1: Keypoints detected in the first image (list of cv2.KeyPoint).
    :param image2: Second image (numpy array) from which keypoints are detected.
    :param keypoints2: Keypoints detected in the second image (list of cv2.KeyPoint).
    :param matches: List of matches between keypoints from the two images (list of cv2.DMatch).
    :param title: Title for the display window (str), defaults to 'Matches'.
    :param scale_percent: Percentage to scale the displayed matches image (int), defaults to 50.
    :return: None. Displays the image with matches drawn and waits for a key press to close.
    '''
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=None,
                       flags=2)

    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, **draw_params)

    # Resize the image to the specified scale percentage
    width = int(img_matches.shape[1] * scale_percent / 100)
    height = int(img_matches.shape[0] * scale_percent / 100)
    img_matches_resized = cv2.resize(img_matches, (width, height))

    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches_resized)
    plt.title(title)
    plt.show()


def draw_matches_ransac(image1, keypoints1, image2, keypoints2, matches, title='Matches', scale_percent=50):
    '''

    Draw matches between two images after applying RANSAC filtering to remove outliers.

    :param image1: First image (numpy array) from which keypoints are detected.
    :param keypoints1: Keypoints detected in the first image (list of cv2.KeyPoint).
    :param image2: Second image (numpy array) from which keypoints are detected.
    :param keypoints2: Keypoints detected in the second image (list of cv2.KeyPoint).
    :param matches: List of matches between keypoints from the two images (list of cv2.DMatch).
    :param title: Title for the display window (str), defaults to 'Matches'.
    :param scale_percent: Percentage to scale the displayed matches image (int), defaults to 50.
    :return: None. Displays the image with matches drawn and waits for a key press to close.
    '''
    ransac_filter = RANSACFilter()  # Create an instance of the RANSAC filter
    M, mask = ransac_filter.apply(keypoints1, keypoints2,
                                  matches)  # Apply RANSAC to get the transformation matrix and mask

    if M is not None:
        # Define parameters for drawing matches with the mask
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=mask.ravel().tolist(),
                           flags=2)

        img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, **draw_params)

        # Resize the image to the specified scale percentage
        width = int(img_matches.shape[1] * scale_percent / 100)
        height = int(img_matches.shape[0] * scale_percent / 100)
        img_matches_resized = cv2.resize(img_matches, (width, height))

        plt.figure(figsize=(10, 5))
        plt.imshow(img_matches_resized)
        plt.title(title)
        plt.show()


def main():
    # Define paths to the images to be processed (B04 channel)
    # image_02_path = 'Alps_dataset_2024/B04/T32TNS_20240215T102029_B04_10m.png'
    # image_03_path = 'Alps_dataset_2024/B04/T32TNS_20240311T101831_B04_10m.png'
    # image_07_path = 'Alps_dataset_2024/B04/T32TNS_20240709T102031_B04_10m.png'
    # image_08_path = 'Alps_dataset_2024/B04/T32TNS_20240828T102021_B04_10m.png'
    # image_10_path = 'Alps_dataset_2024/B04/T32TNS_20241002T101749_B04_10m.png'

    # Define paths to the images to be processed (TCI channel)
    image_02_path = 'Alps_dataset_2024/TCI/T32TNS_20240215T102029_TCI_10m.png'
    image_03_path = 'Alps_dataset_2024/TCI/T32TNS_20240311T101831_TCI_10m.png'
    image_07_path = 'Alps_dataset_2024/TCI/T32TNS_20240709T102031_TCI_10m.png'
    image_08_path = 'Alps_dataset_2024/TCI/T32TNS_20240828T102021_TCI_10m.png'
    image_10_path = 'Alps_dataset_2024/TCI/T32TNS_20241002T101749_TCI_10m.png'

    if not os.path.exists(image_02_path):
        logging.info(f"File does not exist: {image_02_path}")
        exit()
    else:
        logging.info(f"Full path to image1: {os.path.abspath(image_02_path)}")

    if not os.path.exists(image_03_path):
        logging.info(f"File does not exist: {image_03_path}")
        exit()
    else:
        logging.info(f"Full path to image2: {os.path.abspath(image_03_path)}")

    # Load the images using OpenCV
    image1 = cv2.imread(image_02_path)
    image2 = cv2.imread(image_03_path)

    if image1 is None:
        logging.error(f"Error: Could not read image1 from path: {image_02_path}")
        exit()

    if image2 is None:
        logging.error(f"Error: Could not read image2 from path: {image_03_path}")
        exit()

    sift_matcher = SIFTMatcher(ratio_thresh=0.75)
    orb_matcher = ORBMatcher(number_of_points=500)

    sift_performance = measure_performance(sift_matcher, image1, image2)
    orb_performance = measure_performance(orb_matcher, image1, image2)

    logging.info(f"SIFT: Matches - {sift_performance[0]}, Time - {sift_performance[1]:.2f} seconds")
    logging.info(f"ORB: Matches - {orb_performance[0]}, Time - {orb_performance[1]:.2f} seconds")

    # Extract keypoints and descriptors using SIFT
    siftkeypoints1, siftdescriptors1 = sift_matcher.find_keypoints_and_descriptors(image1)
    siftkeypoints2, siftdescriptors2 = sift_matcher.find_keypoints_and_descriptors(image2)

    # Extract keypoints and descriptors using ORB
    orbkeypoints1, orbdescriptors1 = orb_matcher.find_keypoints_and_descriptors(image1)
    orbkeypoints2, orbdescriptors2 = orb_matcher.find_keypoints_and_descriptors(image2)

    # Match descriptors using SIFT and ORB
    good_matches_sift = sift_matcher.match(siftdescriptors1, siftdescriptors2)
    good_matches_orb = orb_matcher.match(orbdescriptors1, orbdescriptors2)

    # Draw matches for SIFT and ORB
    draw_matches(image1, siftkeypoints1, image2, siftkeypoints2, good_matches_sift, 'SIFT Matches')
    draw_matches(image1, orbkeypoints1, image2, orbkeypoints2, good_matches_orb, 'ORB Matches')

    # Draw matches with RANSAC filtering for SIFT and ORB
    draw_matches_ransac(image1, siftkeypoints1, image2, siftkeypoints2, good_matches_sift, 'SIFT Matches with RANSAC')
    draw_matches_ransac(image1, orbkeypoints1, image2, orbkeypoints2, good_matches_orb, 'ORB Matches with RANSAC')


if __name__ == '__main__':
    main()