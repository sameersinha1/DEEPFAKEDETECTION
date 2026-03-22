# Deepfake Image Detection + GAN Training (PyTorch)

This project contains a single script, `deepfake_img.py`, that:

1. Extracts an image dataset from `deepfake.zip`.
2. Trains a CNN discriminator to classify **Real** vs **Fake** images.
3. Evaluates the discriminator with loss, accuracy, and a confusion matrix.
4. Defines and trains a GAN-style generator/discriminator loop.
5. Saves checkpoints for both discriminator-only and GAN training.

## Project Files

- `deepfake_img.py`: End-to-end training and evaluation script.
- `Readme.md`: Project documentation.

## Expected Dataset Structure

The script expects a zip file named `deepfake.zip` in the project root.

After extraction, expected folders are:

```text
dataset/
	deepfake/
		real/
			*.jpg|*.png|...
		Fake/
			*.jpg|*.png|...
```

Notes:
- Folder names are case-sensitive in code: `real` and `Fake`.
- Images are resized to `64x64` during loading.

## What the Script Does

### 1) Setup and Data Preview

- Imports common ML/computer vision libraries (`torch`, `cv2`, `sklearn`, `matplotlib`, etc.).
- Checks CUDA availability.
- Extracts `deepfake.zip` into `dataset`.
- Displays sample images from both classes.

### 2) Discriminator Model

- Defines `Discriminator` (CNN classifier) with:
	- Conv blocks: 3->64->128->256
	- LeakyReLU activations
	- BatchNorm in deeper layers
	- Final linear output (logit)
- Uses `BCEWithLogitsLoss` for binary classification.

### 3) Dataset and DataLoaders

- Builds file path lists:
	- `real` labeled as `1`
	- `Fake` labeled as `0`
- Splits into train/test with `train_test_split(test_size=0.2)`.
- `DeepfakeDataset` reads image with OpenCV, converts BGR->RGB, normalizes to `[-1, 1]`, and returns tensors.

### 4) Discriminator Training

- Trains for `50` epochs using Adam (`lr=0.0002`).
- Saves training loss history.
- Exports weights to `discriminator_model.pth`.

### 5) Discriminator Evaluation

- Loads saved discriminator weights.
- Computes:
	- Test loss
	- Test accuracy
	- Confusion matrix (Fake vs Real)
- Plots training loss curve and confusion matrix.

### 6) GAN Section

- Freezes discriminator parameters once.
- Defines `Generator` that maps `z` (`100`-dim noise) to `3x64x64` images with `Tanh` output.
- Runs GAN training loop for up to `600` epochs.
- Uses gradient clipping for both D and G.
- Saves checkpoint every epoch to `gan_checkpoint.pth`.
- Displays generated image grids every 10 epochs.

## Outputs Produced

- `discriminator_model.pth`: trained discriminator weights.
- `gan_checkpoint.pth`: epoch-wise GAN checkpoint dict with:
	- `epoch`
	- `G`, `D`
	- `optimizer_G`, `optimizer_D`
- Plot windows:
	- Sample dataset images
	- Training loss curve
	- Confusion matrix
	- Generated image grids (periodic)

## Requirements

Install dependencies before running:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib opencv-python scikit-learn ipython
```

If using GPU, install the CUDA-compatible PyTorch build from the official PyTorch selector.

## How to Run

1. Place `deepfake.zip` in the project root.
2. Ensure the zip contains `deepfake/real` and `deepfake/Fake` folders.
3. Run:

```bash
python deepfake_img.py
```

## Current Code Notes

- The script is notebook-style and executes sequentially in one file.
- There is no CLI/config system yet (paths and hyperparameters are hardcoded).
- In the GAN block, make sure generator initialization exists before optimizer creation:

```python
G = Generator().to(device)
```

Without that line, `optimizer_G = optim.Adam(G.parameters(), ...)` will fail.

## Suggested Improvements

- Add argument parsing for dataset path, epochs, and learning rates.
- Split code into modules: data, models, train, evaluate.
- Add reproducibility controls (random seeds, deterministic flags).
- Save generated image snapshots to disk for each checkpoint.
- Add a separate inference script for single-image prediction.
