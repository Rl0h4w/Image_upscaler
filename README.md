# Video Frame Extraction and Super-Resolution

This project involves extracting frames from videos, applying data augmentation and noise, and training a super-resolution model using a combination of EDSR and U-Net architectures.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Main Components](#main-components)
  - [Frame Extraction](#frame-extraction)
  - [Data Augmentation and Standardization](#data-augmentation-and-standardization)
  - [Model Training](#model-training)
  - [Super-Resolution Model](#super-resolution-model)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone <https://github.com/Rl0h4w/Image_upscaler>
    cd <https://github.com/Rl0h4w/Image_upscaler>
    ```

2. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

    Make sure to include the following packages in `requirements.txt` if not already present:
    - tensorflow
    - opencv-python
    - numpy
    - scikit-learn
    - joblib
    - tqdm
    - pytube
    - bs4
    - selenium
    - webdriver-manager

3. **Prepare data directories:**

    ```sh
    mkdir -p data/videos data/frames
    ```

4. **Add YouTube channel URLs:**

    Add the URLs of YouTube channels you want to download videos from into `data/channels_urls.txt`.

## Usage

1. **Download Videos and Extract Frames:**

    The script `loader_videos_YT.py` handles downloading videos from specified YouTube channels and extracting frames from these videos.

    ```sh
    python loader_videos_YT.py
    python main.py
    ```

2. **Train the Super-Resolution Model:**

    The script `main.py` orchestrates the frame extraction, data augmentation, and model training.

    ```sh
    python main.py
    ```

## Project Structure

- `main.py`: Main script for extracting frames, augmenting data, and training the model.
- `models.py`: Contains the model architectures for EDSR and U-Net, and the combined model.
- `loader_videos_YT.py`: Handles downloading videos from YouTube channels.
- `data/`: Directory to store downloaded videos and extracted frames.
- `channels_urls.txt`: Text file containing YouTube channel URLs.

## Main Components

### Frame Extraction

The `extract_frames_from_video` function in `main.py` extracts frames from video files and saves them as PNG images. The `extract_frames` function applies this to all videos in a specified directory using multiprocessing.

### Data Augmentation and Standardization

- `downscale_and_add_noise`: Applies random brightness, saturation, contrast adjustments, and adds noise to an image.
- `data_augmentation`: Defines a Keras Sequential model for additional data augmentation.
- `standardize_frame`: Standardizes the frame using a precomputed scaler.

### Model Training

The `main.py` script prepares the dataset, defines the model, and trains it using the prepared data generators and specified callbacks for checkpoints, early stopping, and learning rate reduction.

### Super-Resolution Model

The model architecture combines EDSR and U-Net:
- `edsr` function in `models.py` defines the EDSR model.
- `unet` function in `models.py` defines the U-Net model.
- `get_combined_model` combines EDSR and U-Net for super-resolution.

## Acknowledgments

- This project uses the following libraries: TensorFlow, OpenCV, NumPy, scikit-learn, Joblib, tqdm, pytube, BeautifulSoup, Selenium, and WebDriver-Manager.
- Special thanks to the developers of these libraries and the open-source community.