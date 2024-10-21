# Satellite Image Matching using SIFT and ORB Algorithms

## Overview

This project demonstrates the application of SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF) algorithms for matching satellite images taken from different seasons. The primary goal is to detect keypoints and match them to analyze similarities and differences between adjacent and opposite seasons using satellite imagery.

## Dataset
The dataset used in this project contains satellite images from the Alps region, captured during different seasons. The Sentinel-2 images download from the official source [Copernicus](https://browser.dataspace.copernicus.eu/)  provided in two channels:

B04 channel (Band 4: Red wavelength)
TCI channel (True Color Image)
Each image has a resolution of 10 meters per pixel. 
The following seasonal pairs are included:

- Winter: February 2024
- Spring: March 2024
- Summer: July 2024, August 2024
- Autumn: October 2024

## Algorithms
The project applies the following algorithms:

**SIFT (Scale-Invariant Feature Transform)** — Detects keypoints and descriptors invariant to scale and rotation, allowing robust image matching across different conditions.
**ORB (Oriented FAST and Rotated BRIEF)** — A more computationally efficient alternative to SIFT, designed to perform similarly while being faster and using less memory.

**RANSAC (Random Sample Consensus)** is applied after keypoint matching to remove outliers and refine the transformation matrix, ensuring only accurate matches are considered.


## Project Structure

- `images_dataset.ipynb`: Jupyter notebook used to load, preprocess, and visualize the satellite images dataset.
- `algorithms.py`: Python module that contains the implementations of SIFT and ORB keypoint detection and matching algorithms, as well as performance measurement and RANSAC filtering.
- `matching_inference.py`: Python script that demonstrates keypoint detection and matching process using SIFT and ORB with and without RANSAC filtering, comparing different satellite images across seasons.
- `sentinel-2_image_matching_demo.ipynb`: A Jupyter notebook providing a detailed step-by-step demonstration of the keypoint detection, matching, and RANSAC filtering process applied to Sentinel-2 satellite images.
- `README.md`: Documentation of the project.
- `requirements.txt`: File containing the list of dependencies and libraries required to run the project.

## How to Run the Project

1. **Clone the Repository**
    - Start by cloning the project repository to your local machine:
      ```bash
      git clone https://github.com/HannaVasiunyk/Sentinel-2-image-matching.git
      cd Sentinel-2-image-matching
      ```

2. **Set Up a Virtual Environment (Optional)**
    - It's recommended to create a virtual environment to avoid conflicts with other projects:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      ```

3. **Install Dependencies**
    - Install the required libraries from the `requirements.txt` file:
      ```bash
      pip install -r requirements.txt
      ```
4. **Upload raw images:**

- Upload raw images [here](https://drive.google.com/drive/folders/1lbpE4eCbMO_suSGSekwDygn0wVEJpHCL?usp=drive_link).

5. **Run the images dataset notebook:**

- Open `images_dataset.ipynb` in Jupyter Notebook and execute the cells to create the dataset.

6.  **Check the inference script:**

- Run `matching_inference.py` Run the matching_inference.py script to demonstrate keypoint detection and matching using SIFT and ORB with RANSAC filtering:
   ```bash
     python matching_inference.py
   ```
  
7. **Check the demo notebook:**

- Open `sentinel-2_image_matching_demo.ipynb` in Jupyter Notebook to review the satellite images matching adjacent and opposite seasons.