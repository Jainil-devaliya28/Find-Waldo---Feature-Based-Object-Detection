
Image Processing and Waldo Detection

A Python-based computer vision project exploring image filtering, template matching, feature detection, and feature-based object localization. The project implements fundamental computer vision algorithms using NumPy, OpenCV, and Matplotlib, with an emphasis on understanding the underlying mathematics.

## Overview

This project is organized into four tasks, progressing from basic image processing to feature-based object detection.

### Task 1: Image Filtering and Processing
- Implemented 2D convolution with appropriate boundary handling.
- Implemented Gaussian blur using separable one-dimensional Gaussian kernels.
- Converted RGB images to grayscale using luminance-weighted channel contributions.
- Generated image-processing visualizations to examine filtering and edge-related results.

### Task 2: Template Matching
- Implemented integral images (summed-area tables) for efficient rectangular region summation.
- Implemented zero-mean normalized cross-correlation (ZNCC/NCC) for matching image templates.
- Explored multi-scale template matching to locate Waldo at different sizes.
- Evaluated localization results using Intersection over Union (IoU).

### Task 3: Feature Detection and Description
- Implemented the Harris corner detector to identify distinctive image corners.
- Implemented Difference of Gaussians (DoG) keypoint detection across multiple scales.
- Explored gradient-based feature representations and orientation histograms.
- Visualized detected features to analyze their distribution and usefulness for matching.

### Task 4: Feature Matching and Object Localization
- Implemented descriptor-distance computation using Euclidean distance.
- Applied Lowe's ratio test to filter ambiguous feature matches.
- Used mutual nearest-neighbor matching to improve correspondence reliability.
- Combined feature correspondences through consensus voting to estimate Waldo's location.
- Evaluated matching behavior across scenes with different difficulty levels and rotations.

## Technologies Used

- **Python 3.9+**
- **NumPy** — numerical computation and array operations
- **OpenCV** — image loading, color conversion, and resizing
- **Matplotlib** — visualization of images, features, and matching results

## Project Structure

```text
23110090_Assignment2/
├── task1.py
├── task2.py
├── task3.py
├── task4.py
├── requirements.txt
├── subjective_answers_task1.md
├── subjective_answers_task2.md
├── subjective_answers_task3.md
├── subjective_answers_task4.md
├── imgs/
│   ├── waldo_template.png
│   ├── scene_easy.png
│   ├── scene_medium.png
│   ├── scene_hard.png
│   ├── scene_rotated.png
│   └── annotations.json
└── plots/
    ├── task1_results.png
    ├── task1_edges.png
    ├── task2_results.png
    ├── task3_results.png
    └── task4_*.png
```

*Note: The directory structure above is illustrative. Update the filenames to match the actual files in your repository.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd 23110090_Assignment2
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual repository URL.

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, create one with the dependencies used by your project:

```text
numpy
opencv-python
matplotlib
```

### 4. Verify the Installation

```bash
python -c "import cv2, numpy, matplotlib; print('Environment ready!')"
```

## Running the Project

Run the scripts from the project root directory:

```bash
python task1.py
python task2.py
python task3.py
python task4.py
```

The scripts generate plots and visualizations in the output directory, depending on their implementation. Run the tasks in order if later scripts depend on functions from earlier tasks.

## Results and Visualizations

The project includes visual outputs illustrating the behavior of the implemented algorithms.

- **Task 1:** Image filtering and edge-related visualizations.
- **Task 2:** Template-matching results for Waldo localization.
- **Task 3:** Feature detection and keypoint visualizations.
- **Task 4:** Feature correspondences and consensus-based localization across multiple scenes.

The `plots/` directory contains generated output images, where available. Review these visualizations alongside the corresponding Python implementations and written analysis to understand the algorithms' behavior and limitations.

## Key Computer Vision Concepts

| Concept | Purpose |
|---|---|
| 2D convolution | Apply spatial filters to images |
| Gaussian filtering | Smooth images and reduce high-frequency noise |
| Integral images | Calculate rectangular sums efficiently |
| Normalized cross-correlation | Measure template similarity |
| Harris corner detection | Identify distinctive corners |
| Difference of Gaussians | Detect keypoints at multiple scales |
| Feature descriptors | Represent local image structures |
| Ratio test | Reject ambiguous feature matches |
| Mutual nearest neighbors | Improve matching consistency |
| Consensus voting | Estimate object location from feature correspondences |
| Intersection over Union | Evaluate bounding-box localization |

## Learning Outcomes

Through this project, I explored:

- The mathematical foundations of image filtering and convolution.
- Efficient image-processing techniques using NumPy array operations.
- The strengths and limitations of template matching under scale variations.
- Corner and scale-space feature detection.
- Descriptor matching and rejection of unreliable correspondences.
- Object localization through geometric consensus and quantitative evaluation.
- The importance of visualizing intermediate results when debugging computer vision algorithms.


