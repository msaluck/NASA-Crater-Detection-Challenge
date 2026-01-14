# NASA Crater Detection Baseline

This project contains a baseline solution for the NASA Crater Detection Challenge. It uses classical computer vision techniques (OpenCV) to detect craters in satellite imagery and fits ellipses to them.

## Project Structure

*   `detect.py`: Core detection logic using Canny Edge Detection and Hough Transforms.
*   `run_local.py`: Main execution script to process the dataset and generate a solution file.
*   `scorer.py`: Evaluation script to calculate the score of your solution against the ground truth.
*   `geometry_filter.py` & `ellipse_utils.py`: Helper utilities for processing geometric shapes.

## Prerequisites

*   Python 3.x
*   Data folder structure located at `../nasa-craters-data/` relative to this folder.

## Installation

1.  Navigate to the project directory:
    ```bash
    cd crater_baseline
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Generating Detections

To run the detection algorithm on the training set:

```bash
python run_local.py
```

This script reads images from `../nasa-craters-data/train/`, processes them, and saves the detected craters to `output/solution.csv`.

*Note: You may need to adjust `IMG_DIR` in `run_local.py` if your data is located elsewhere.*

### 2. Scoring

To evaluate the performance of your solution locally:

```bash
python scorer.py --pred output/solution.csv --truth ../nasa-craters-data/train-gt.csv --out_dir output/
```

This will compare your `solution.csv` against the ground truth labels in `train-gt.csv` and output the results.

## Output Format

The output `solution.csv` follows the challenge submission format with columns:
*   `ellipseCenterX(px)`
*   `ellipseCenterY(px)`
*   `ellipseSemimajor(px)`
*   `ellipseSemiminor(px)`
*   `ellipseRotation(deg)`
*   `crater_classification`
