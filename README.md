
# Synthetic CAPTCHA Generator for AI/ML Datasets

A powerful, highly-configurable Python script for generating synthetic CAPTCHA images. This tool is designed to create robust, large-scale datasets for training machine learning and computer vision models.

This script can be used to reverse-engineer and replicate complex visual styles, including multi-layer noise, advanced text distortion, and procedural artifacts.

**Author:** Kalana Gayan Senevirathne

## Key Features

* **Full Dataset Generation:** Automatically generate thousands of images with `train/val/test` splits and a corresponding `labels.csv` file, ready for any ML pipeline.
* **Highly Configurable:** Nearly every visual parameter is exposed in the script's settings block, allowing for precise control over fonts, colors, noise, and distortions.
* **Advanced Text Distortion:**
    * Word-level sine wave distortion.
    * Per-character pixel warping, rotation, and shearing.
    * Precise control over character spacing and positioning.
* **Multi-Layer Noise Engine:**
    * Procedural 2x2 "cluster dot" noise with configurable density and color palettes.
    * Semi-transparent, randomly angled line noise.
* **Built with a Pro-Stack:** Uses a fast and efficient stack of **OpenCV**, **Pillow (PIL)**, and **NumPy** for all image manipulations.

## Example Output

The generator can be configured to produce a wide variety of styles. Here are a few examples created by the script:


<img width="460" height="115" alt="007-TAO" src="https://github.com/user-attachments/assets/469a40d6-fdb8-4271-8ddb-4df6e4c462ab" />
<img width="460" height="115" alt="006-SKL" src="https://github.com/user-attachments/assets/04a1318d-99f2-4c5c-ad76-30ca5eea5bca" />
<img width="460" height="115" alt="005-MAV" src="https://github.com/user-attachments/assets/a63e5381-18ae-4af7-8c05-36f21d13d8a6" />
<img width="460" height="115" alt="008-STOP" src="https://github.com/user-attachments/assets/d36d9b3d-cf8c-4708-8ec8-dc419004a748" />
<img width="460" height="115" alt="009-SPOT" src="https://github.com/user-attachments/assets/d323a672-6d7e-44e9-ae3f-273a7c8f6123" />
<img width="460" height="115" alt="010-W" src="https://github.com/user-attachments/assets/877dc15f-0f7d-4de3-bff8-5f8c46e80740" />

## Installation

This project requires Python 3 and the following libraries.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  Install the required libraries:
    ```bash
    pip install opencv-python pillow numpy tqdm
    ```

3.  (Optional) Create a `requirements.txt` file with the following:
    ```
    opencv-python
    Pillow
    numpy
    tqdm
    ```
    And install with: `pip install -r requirements.txt`

## Usage

The script is run from the command line and has two primary modes.

### 1. Generating a Full Dataset (Recommended)

To generate a complete dataset (e.g., 1000 images), use the `--total` argument.

```bash
python captcha_generator.py --total 1000
```
This will:

    - Create a captcha_dataset directory.

    - Automatically split the 1000 images into train, val, and test sub-folders.

    - Generate a labels.csv file in the root, mapping each image file to its correct label.

The labels used for generation can be edited in the DEFAULT_LABELS_LIST inside the script.

### 2. Generating a Single Test Image

To quickly test a single word, use the --single argument.
```Bash
python captcha_generator.py --single "FARTCOIN"
```
This will create a single image named test_FARTCOIN.png in the project's root directory.

### Configuration

All visual parameters can be changed by editing the ⚙️ FINAL SETTINGS block at the top of captcha_generator.py.

This includes:
    - Canvas size
    - Font files (.ttf) and sizes
    - Text colors
    - All distortion, wave, and shear amplitudes
    - Line noise properties (count, color, width)
    - Cluster dot noise properties (density, colors)
    - Final blur and rotation effects

Tweak these variables to create entirely new CAPTCHA styles.

### License

This project is open-source. Please feel free to use, modify, and distribute. (Consider adding an MIT License file to your repository).
